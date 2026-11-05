import argparse
import gc
import os
import types

import torch
import torch.optim as optim

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

from augmult.augmentations import get_transforms_from_str
from train import train, eval, default_metric_name
from augmult.data import dp_dataloader, init_mv_collate, non_dp_tokenize_dataloader, get_dataset, has_labels
from util.early_stopper import EarlyStopping
from util.privacy_engine_util import _prepare_model_modified, empty_batch_handling, prepare_gradsamplers, add_noise
from util.logging_util import aug_name, load_ckpt, save_ckpt
from util.different_finetune_modes import get_param_setter_from_str, model_and_tokenizer
from util.reproducibility import set_seed

import wandb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')


def wrap_run_cuda(params):
    try:
        score = run(**params)
    except Exception as e:
        try:
            wandb.config.update({"failed_because": repr(e)}, allow_val_change=True)
        except Exception:
            pass
        print("\n\n---------------\nError\n--------\n" + repr(e))
        raise

    torch.cuda.empty_cache()
    gc.collect()
    return score


def resolve_delta(spec, n):
    """delta spec -> float. None / "1/n" -> 1/n (thesis default); "n^-1.1" -> n**-1.1; else a float."""
    if spec is None:
        return 1 / n
    if isinstance(spec, (int, float)):
        return float(spec)
    s = str(spec).replace(" ", "")
    if s == "1/n":
        return 1 / n
    if s.startswith("n^"):
        return n ** float(s[2:])
    if s.startswith("1/n^"):
        return n ** -float(s[4:])
    return float(s)


def _trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _restore_snapshot(model, snapshot):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in snapshot.items():
            params[n].copy_(v)


def run(dataset_name, transform_list, K, epochs, batch_size, lr, max_grad_norm, trainable_param_setter, num_labels,
        noise_multiplier=None, dataset_size=None, save_model=None, target_epsilon=None, early_stop_patience=10,
        model_name = "bert-base-uncased", glue=True, max_phys_batchsize=1500,total_steps=None,
        seed=None, delta=None, accountant="rdp", restore_best=False, eval_test=True,
        multi_label=False, metric=None,
        hf_path=None, hf_config=None, text_columns=None, label_column="label", splits=None):

    # ---------------- Initialisation ------------------
    # Seeding (seed=None keeps the unseeded thesis behaviour)
    generator = set_seed(seed)

    # Model, Tokenizer
    model, tokenizer = model_and_tokenizer(model_name, num_labels, multi_label=multi_label)
    total_p, trainable_p = trainable_param_setter(model)
    finetune_percent = trainable_p / total_p * 100

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Dataloaders
    precomputed = None in transform_list
    precomputed_views = getattr(transform_list, "precomputed_views", None)
    dataset = get_dataset(dataset_name, glue = glue, precomputed_augs=precomputed, precomputed_views=precomputed_views,
                          hf_path=hf_path, hf_config=hf_config, text_columns=text_columns,
                          label_column=label_column, splits=splits)
    mv_train_loader = dp_dataloader(dataset["train"],dataset_size,tokenizer,transform_list,batch_size,generator=generator)
    valid_loader = non_dp_tokenize_dataloader(dataset['validation'], tokenizer, max_phys_batchsize)

    steps_per_epoch = len(mv_train_loader)
    if epochs is None:
        if total_steps is None: raise ValueError("Specify either epochs or total_steps")
        epochs = -(total_steps// -steps_per_epoch)
        early_stop_patience = 1000 #max(5,(epochs//2))

    # Logging
    n_samples = len(mv_train_loader.dataset)
    sampling_rate = batch_size / n_samples
    metric_name = metric or default_metric_name(dataset_name, multi_label)

    wandb.config.update({"early_stop_patience":early_stop_patience,"epochs":epochs,'K': K,'batches per epoch': steps_per_epoch,
        "transform_list": aug_name(transform_list),"precomputed_views": precomputed_views,
        "finetune_percent":finetune_percent,"sampling_rate":sampling_rate,
        "seed": seed, "accountant": accountant, "metric": metric_name, "restore_best": restore_best,}, allow_val_change=True)

    # ------------- Make DP with AugMult -----------------

    delta_value = resolve_delta(delta, n_samples)
    wandb.config.update({"delta_spec": "1/n" if delta is None else str(delta), "delta": delta_value}, allow_val_change=True)
    early_stop = EarlyStopping(patience=early_stop_patience)

    # Ensure DP compatibility
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine(accountant=accountant)

    # Hack to load the custom AugmultGradSamplerModule
    privacy_engine._prepare_model = types.MethodType(_prepare_model_modified, privacy_engine)

    # Make Private
    make_private_params = {'module': model,'optimizer': optimizer,'data_loader': mv_train_loader,'max_grad_norm': max_grad_norm,'grad_sample_mode': "augmult",}
    if target_epsilon is None:
        dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private(**make_private_params,noise_multiplier=noise_multiplier,)
    else:
        dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(**make_private_params,target_epsilon=target_epsilon,target_delta=delta_value,epochs=epochs)

    resume_ckpt = os.environ.get("CKPT_ID",None)
    ckpt_dict = load_ckpt(privacy_engine,dp_model,resume_ckpt) if resume_ckpt is not None else {}

    # Hack to modify Optimizer
    dp_optimizer.add_noise = types.MethodType(add_noise, dp_optimizer)

    wandb.config.update({"noise_multiplier": dp_optimizer.noise_multiplier}, allow_val_change=True)

    # Override the empty batch shapes provided by privacy engine
    mv_collate = init_mv_collate(tokenizer,transform_list,max_length=128,precomputed=precomputed)
    empty_batch_handling(mv_train_loader=mv_train_loader,dp_train_loader=dp_train_loader,mv_collate=mv_collate)

    # Custom Grad Samplers
    dp_model = prepare_gradsamplers(K,dp_model)

    # -------------- Training ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_model.to(device=device)

    start_epoch=ckpt_dict.get("epoch",0) # because logging starts counting at 1 not 0, this is correct
    steps = steps_per_epoch*start_epoch
    train_inputs = {
        'dp_model': dp_model,
        'task_name':dataset_name,
        'dp_train_loader': dp_train_loader,
        'dp_optimizer': dp_optimizer,
        'device': device,
        'K': K,
        'max_phys_batch_size': max_phys_batchsize,
        'steps': steps, # possibly overwrite some logged steps when requeueing (real batchsize unknown)
        'multi_label': multi_label,
        'metric': metric,
    }
    eval_kwargs = {'device': device, 'task_name': dataset_name, 'multi_label': multi_label, 'metric': metric}

    valid_metric, valid_loss = 0, float('nan')
    best_valid, best_epoch, best_state = None, None, None
    for i in range(start_epoch,epochs):
        # Train for 1 epoch
        train_inputs['steps'] = train(**train_inputs)
        # Evaluate
        valid_metric, valid_loss = eval(dp_model, valid_loader, **eval_kwargs)
        wandb.log({"epoch": i+1,"valid_metric": valid_metric,"valid_loss": valid_loss})
        if restore_best and (best_valid is None or valid_metric > best_valid):
            best_valid, best_epoch, best_state = valid_metric, i+1, _trainable_snapshot(dp_model)
        # early stopping
        if early_stop.step(valid_loss):
            wandb.config.update({"Stopped early": True}, allow_val_change=True)
            break
        # save ckpt
        if save_model:
            d = wandb.config.as_dict()
            d["epoch"]=i+1
            save_ckpt(ckpt_dict=d,privacy_engine=privacy_engine,model=dp_model,filename=f"{wandb.run.id}")#f"{save_model}_{wandb.run.name}")

    # ------------ Evaluation ---------------------

    # Privacy (training data only; validation-based model selection is not accounted for)
    try:
        real_eps = privacy_engine.accountant.get_epsilon(delta=delta_value)
    except OverflowError:
        real_eps = float('inf')
    wandb.log({"epsilon": real_eps, "delta": delta_value})

    # Optionally restore the best-validation epoch before testing (off by default = thesis behaviour: last epoch)
    if restore_best and best_state is not None:
        _restore_snapshot(dp_model, best_state)
        wandb.log({"best_epoch": best_epoch, "best_valid_metric": best_valid})

    # Test split
    test_split = dataset.get('test')
    if eval_test and test_split is not None and has_labels(test_split):
        test_loader = non_dp_tokenize_dataloader(test_split, tokenizer, max_phys_batchsize)
        test_metric, test_loss = eval(dp_model, test_loader, **eval_kwargs)
        wandb.log({"test_metric": test_metric, "test_loss": test_loss})
        print(f"test {metric_name}: {test_metric:.4f} (loss {test_loss:.4f})")
    elif eval_test:
        print("No labelled test split (e.g. GLUE hides test labels); skipping test evaluation.")
        wandb.config.update({"test_skipped": "no labelled test split"}, allow_val_change=True)

    print(f"valid {metric_name}: {valid_metric:.4f} (loss {valid_loss:.6f}) | epsilon {real_eps:.3f} at delta {delta_value:.3g} ({accountant})")
    if valid_metric: return valid_metric # returning for sweep setup


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {v!r}")


def _parse_splits(s):
    # "train=train,validation=validation_matched,test=test_matched"
    if not s:
        return None
    return dict(kv.split("=", 1) for kv in s.split(","))


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation Multiplicity for DP-NLP Models")

    # Training Args
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (ignored if --total_steps is set)')
    parser.add_argument('--total_steps', type=int, default=None, help='Train for ceil(total_steps/steps_per_epoch) epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in dataset')
    parser.add_argument('--multi_label', action='store_true', help='Multi-label classification (sigmoid/BCE, micro-F1), e.g. HoC with --num_labels 10')
    parser.add_argument('--metric', type=str, default=None, choices=['accuracy','mcc','f1_weighted','micro_f1','macro_f1'], help='Override the task default metric')
    parser.add_argument('--dataset_size', type=int, default=None, help='Dataset size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (python/numpy/torch, data sampling, DP noise)')

    # DP
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--noise_multiplier', type=float, default=None, help='Noise multiplier for DP training (overrides --target_epsilon)')
    parser.add_argument('--target_epsilon', type=float, default=32, help='Target epsilon for differential privacy')
    parser.add_argument('--delta', type=str, default="1/n", help='delta: "1/n" (thesis default), "n^-1.1", or a float')
    parser.add_argument('--accountant', type=str, default="rdp", choices=["rdp", "prv"], help='Privacy accountant')
    parser.add_argument('--transforms_name', type=str, default=None, help='Augmentations: None, identityK, edaK, precomputed_AxB, precomputed:<view,...>[*B]')

    # Logging and setup etc
    parser.add_argument('--save_model', type=str, default=None, help='Whether to save the model')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_phys_batchsize', type=int, default=1500, help='how many samples fit on the device per step')
    parser.add_argument('--param_setter', type=str, default="classifier_and_pooler", help='Which parameters to fine-tune')
    parser.add_argument('--restore_best', action='store_true', help='Restore the best-validation epoch before the test evaluation')
    parser.add_argument('--skip_test', action='store_true', help='Do not evaluate the test split')

    # Model and Dataset
    parser.add_argument('--experiment_name', type=str, default="untitled", help='Name of the experiment')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased", help='Model name to use')
    parser.add_argument('--dataset_name', type=str, default="sst2", help='Dataset name (GLUE task, local dataset, or a label for --hf_path)')
    parser.add_argument('--glue', type=str2bool, default=True, help='Whether the dataset is from HF glue (true/false)')
    parser.add_argument('--hf_path', type=str, default=None, help='Any HF hub dataset (implies --glue false), e.g. stanfordnlp/snli')
    parser.add_argument('--hf_config', type=str, default=None, help='HF dataset config name')
    parser.add_argument('--text_columns', type=str, default=None, help='Comma-separated text column(s), e.g. premise,hypothesis')
    parser.add_argument('--label_column', type=str, default="label", help='Label column name')
    parser.add_argument('--splits', type=str, default=None, help='Split mapping, e.g. train=train,validation=validation_matched,test=test_matched')

    # Parse the arguments
    args = parser.parse_args()
    if args.hf_path is not None:
        args.glue = False
    if args.noise_multiplier is not None and args.target_epsilon is not None:
        print("--noise_multiplier given: ignoring --target_epsilon")
        args.target_epsilon = None
    if args.total_steps is not None:
        args.epochs = None

    return args


def commandline():
    args = parse_args()

    # ------------- Logging ----------------------
    wandb.init(project=args.dataset_name, name=args.experiment_name)

    wandb.config.update(vars(args))

    trainable_param_setter = get_param_setter_from_str(args.param_setter)
    transform_list = get_transforms_from_str(args.transforms_name)
    K = len(transform_list)


    # Call the function with parsed arguments
    run(
        dataset_name=args.dataset_name,
        transform_list=transform_list,
        K=K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        trainable_param_setter=trainable_param_setter,
        num_labels=args.num_labels,
        noise_multiplier=args.noise_multiplier,
        dataset_size=args.dataset_size,
        save_model=args.save_model,
        target_epsilon=args.target_epsilon,
        early_stop_patience=args.early_stop_patience,
        model_name=args.model_name,
        glue=args.glue,
        max_phys_batchsize=args.max_phys_batchsize,
        total_steps=args.total_steps,
        seed=args.seed,
        delta=args.delta,
        accountant=args.accountant,
        restore_best=args.restore_best,
        eval_test=not args.skip_test,
        multi_label=args.multi_label,
        metric=args.metric,
        hf_path=args.hf_path,
        hf_config=args.hf_config,
        text_columns=args.text_columns.split(",") if args.text_columns else None,
        label_column=args.label_column,
        splits=_parse_splits(args.splits),
    )

    wandb.finish()


def debug(params):

    wandb.init(project=params['dataset_name'], name="debug")
    wandb.config.update(params)

    trainable_param_setter = get_param_setter_from_str(params.pop("param_setter", "classifier_and_pooler"))
    transform_list = get_transforms_from_str(params["transform_name"])
    del params["transform_name"]
    K = len(transform_list)

    params.update({"K":K,"transform_list":transform_list,"trainable_param_setter":trainable_param_setter})

    run(**params)


if __name__ == "__main__":
    commandline()
