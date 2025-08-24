from libs import *
def download_kaggle_dataset(url):
    # Download latest version
    path = kagglehub.dataset_download(url)
    print("Path to dataset files:", path)
    # /Users/jing/.cache/kagglehub/datasets/alxmamaev/flowers-recognition/versions/2/flowers/
    return path
# download_kaggle_dataset("alxmamaev/flowers-recognition")

def load_data(data_dir):
    print('--------- generate dataset ---------')
    # create customized transforms:
    #   handle different image size
    #   PIL image to tensor
    #   normalize to speed up training, remember to use same normalization during eval/test
    #      use the normalization factor from a pre-trained model if you want to leverage the model in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    print('dataset length: {}'.format(len(dataset)))
    print('dataset classes: ', dataset.classes)
    print('image shape: ', dataset[0][0].shape, 'label: ', dataset.classes[dataset[0][1]])
    show_image(unnormalize(dataset[0][0]))

    return dataset

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    print(tensor.shape)
    tensor = tensor * std + mean
    return tensor

def show_image(image):
    plt.imshow(image.permute(1, 2, 0))
    # plt.axis('off')
    plt.show()

def show_batch(image_dataloader):
    for images, labels in image_dataloader:
        images = unnormalize(images)
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        plt.show()
        break


def record_results(args, history, filename='training_log.txt', duration=0):
    # Create the file path
    log_dir = os.path.join(os.getcwd(), 'log')
    log_path = os.path.join(log_dir, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print('Created new log directory: {}'.format(log_dir))

    with open(log_path, 'a') as f:
        f.write("--- Training Run ---\n")
        f.write(f"Timestamp: {time.ctime()}\n")

        # Write arguments
        f.write("Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")

        # Write final results
        f.write("\n\nEvaluation Results:\n")
        f.write('\n'.join(history))
        f.write("\n\nTraining duration: {}\n".format(duration))
        f.write("\n")
