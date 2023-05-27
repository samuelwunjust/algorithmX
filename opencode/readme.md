2023.5.11 algorithmX now support load data(FashionMinist) some common neural network :VGG,AlexNet,GoogleLeNet,ResNet DenseNet the train function &the evaulate_accuracy function

you can use algorithmX such as 

```{.python .input}
import algorithmX as alx
train_iter,test_iter=alx.load_data()
num_epochs=10,lr=0.01,device=("cuda")
alx.train(train_iter,test_iter,num_epochs,lr,device)
```

2023.5.27
update the train_and_evaluate function with tqdm function
