from data import get_dataloaders

if __name__ == '__main__' :

    train_loader, test_loader = get_dataloaders('./', batch_size=32, num_workers=4, pin_memory=True)

    for sample in train_loader :
        X, y = sample
        print(X.shape, y.shape)
        break