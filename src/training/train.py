import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        positive_labels = d["positive_labels"].to(device)
        negative_labels = d["negative_labels"].to(device)
        true_weights = d["weight"].to(device)
        true_label_types = d["label_type"].to(device)

        positive_outputs, negative_outputs, weight_outputs, label_type_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(positive_outputs, negative_outputs, weight_outputs, label_type_outputs,
                       positive_labels, negative_labels, true_weights, true_label_types)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            positive_labels = d["positive_labels"].to(device)
            negative_labels = d["negative_labels"].to(device)
            weights = d["weight"].to(device)
            label_types = d["label_type"].to(device)

            positive_outputs, negative_outputs, weight_outputs, label_type_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(positive_outputs, negative_outputs, weight_outputs, label_type_outputs,
                           positive_labels, negative_labels, weights, label_types)

            losses.append(loss.item())

    return np.mean(losses)

def train_model(model, train_loader, loss_fn, optimizer, scheduler, device, epochs):
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_loader.dataset)
        )

        print(f'Train loss {train_loss}')

    return model