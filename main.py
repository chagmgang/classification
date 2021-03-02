import torch
import torchvision
import datasets
import model
import tqdm

def run_epoch(net, criterion, optimizer, dataloader, device, is_train):

    pred_list = []
    loss_list = []

    if is_train:
        model = model.train()
    else:
        model = model.eval()

    for data in tqdm.tqdm(dataloader):

        image, target = data
        image = image.to(device)
        target = target.to(device)

        if is_train:
            optimizer.zero_grad()
            logit = net(image)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                logit = net(image)
                loss = criterion(logit, target)

        pred = torch.argmax(logit, axis=1)
        pred = pred.eq(target)

        loss_list.append(loss.item())
        pred_list.extend(pred.item())

    loss_mean = torch.mean(torch.stack(loss_list))
    pred_mean = torch.mean(torch.stack(pred_list))

    return loss_mean, pred_mean

def main():

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    train_dataloader = datasets.make_dataset(is_train=True, batch_size=32)
    valid_dataloader = datasets.make_dataset(is_train=False, batch_size=32)

    net = model.Model(num_class=100)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    epochs = 30

    for epoch in range(epochs):

        train_loss, train_pred = run_epoch(
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                dataloader=train_dataloader,
                is_train=True,
                device=device)

        valid_loss, valid_pred = run_epoch(
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                dataloader=valid_dataloader,
                is_train=False,
                device=device)

        print('----------------------')
        print(f'train loss : {train_loss}')
        print(f'train pred : {train_pred}')
        print(f'valid loss : {valid_loss}')
        print(f'valid pred : {valid_pred}')
        print('----------------------')

if __name__ == '__main__':
    main()
