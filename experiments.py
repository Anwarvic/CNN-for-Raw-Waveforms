from models import *
from data_loader import load_data
from train_test import train
from utils import *



if __name__ == "__main__":
    random_seed= 42
    shuffle_dataset = True
    model = m3
    batch_size = 128
    num_epochs = 2
    data_path = "urbansound8k/"

    # Load data
    train_loader, test_loader = load_data(data_path, batch_size, shuffle_dataset, random_seed)

    # apply initializer
    model.apply(init_weights)
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))
    
    # create criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4) #L2 regularization is added
    
    accs = train(model, criterion, train_loader, test_loader, optimizer, num_epochs=num_epochs)

    draw_graph(accs)


