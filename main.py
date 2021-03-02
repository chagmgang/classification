import torch
import torchvision
import datasets
import model
import tqdm

def run_epoch(net, criterion, optimizer, dataloader, device, is_train):

    pred_list = []
    loss_list = []

    if is_train:
        net = net.train()
    else:
        net = net.train()

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

        loss = loss.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        loss_list.append(loss)
        pred_list.extend(pred)

    loss_mean = sum(loss_list) / len(loss_list)
    pred_mean = sum(pred_list) / len(pred_list)

    return loss_mean, pred_mean

def main():

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    train_dataloader = datasets.make_dataset(is_train=True, batch_size=128)
    valid_dataloader = datasets.make_dataset(is_train=False, batch_size=128)

    net = model.Model(num_class=100)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    epochs = 100

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
