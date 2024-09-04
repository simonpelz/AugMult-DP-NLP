from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

def classifier_only(model):
    layers = [model.classifier]
    return set_from_layer_list(layers,model)


def last_3(model):
    layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
    return set_from_layer_list(layers,model)


def classifier_and_pooler(model):
    layers = [model.bert.pooler, model.classifier]
    return set_from_layer_list(layers,model)


def bias_and_classifier(model):
    total_params,trainable_params = 0,0
    for name,p in model.named_parameters():
        if "bias" in name or "classifier" in name:
            p.requires_grad = True
            trainable_params += p.numel()
        else:
            p.requires_grad = False
        total_params += p.numel()
    return total_params,trainable_params 


def crosscheck_bias(model):
    total_params,trainable_params = 0,0
    for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

    for n,p in model.bert.pooler.named_parameters():
        if "bias" in n:
            p.requires_grad = True
            trainable_params += p.numel()
    for p in model.classifier.parameters():
        p.requires_grad = True
        trainable_params += p.numel()
    
    return total_params,trainable_params 


def set_from_layer_list(layer_list, model):
    total_params,trainable_params = 0,0
    for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

    for layer in layer_list:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    return total_params, trainable_params


def model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model, tokenizer

def main():
    MODEL_NAME = "bert-base-uncased"
    NUM_LABELS = 2

    model, _ = model_and_tokenizer(MODEL_NAME, NUM_LABELS)

    #layer_list = [model.bert.pooler, model.classifier]
    #trainable_param_setter = lambda model: set_from_layer_list(layer_list,model)
    trainable_param_setter = classifier_only
    total_p, trainable_p = trainable_param_setter(model)

    print(f"total params: {total_p}, trainable:{trainable_p}")
    print(f"finetune percentage: {(trainable_p/total_p)*100:.4f}%")



if __name__ == "__main__":
    main()