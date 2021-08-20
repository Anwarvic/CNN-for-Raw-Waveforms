import argparse

from models import *
from data_loader import load_urbansound8k
from train_test import train
from utils import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--model', default="M3", type=str, help='The model name')
    parser.add_argument('--n', default=2, type=int, help='Max number of epochs')
    parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle Training data?')
    parser.add_argument('--seed', default=42, type=int, help='Random seed to use')
    parser.add_argument('--batchSize', default=128, type=int, help='Batch size to use for train/test sets')
    parser.add_argument('--dataPath', default="urbansound8k", type=str, help="Relative path of the dataset")

    args=parser.parse_args()
    if args.model == "M3":
        model = m3
    elif args.model == "M5":
        model = m5
    elif args.model == "M11":
        model = m11
    elif args.model == "M18":
        model = m18
    elif args.model == "M34_res":
        model = m34_res
    random_seed= args.seed
    shuffle_dataset = args.shuffle
    batch_size = args.batchSize
    num_epochs = args.n
    data_path = args.dataPath

    # Load data
    train_loader, test_loader = load_urbansound8k(data_path, batch_size, shuffle_dataset, random_seed)

    # apply initializer
    model.apply(init_weights)
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))
    
    # create criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4) #L2 regularization is added
    
    accs = train(model, criterion, train_loader, test_loader, optimizer, num_epochs=num_epochs)

    draw_graph(accs)
