import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report


def write_report(y_test, predictions, mlb, file_path, save=True):
    class_report = classification_report(y_test.argmax(axis=1),
                                         predictions.argmax(axis=1),
                                         target_names=[str(x) for x in mlb.classes_])
    if save:
        with open(file_path, "w+") as f:
            f.write(class_report)
    return class_report


def draw_graph(model_history, epoch_num, file_path):
    plt.style.use("ggplot")
    plt.figure()
    try:
        plt.plot(numpy.arange(0, epoch_num), model_history.history["loss"], label="train_loss")
        plt.plot(numpy.arange(0, epoch_num), model_history.history["val_loss"], label="val_loss")
        plt.plot(numpy.arange(0, epoch_num), model_history.history["categorical_accuracy"], label="train_acc")
        plt.plot(numpy.arange(0, epoch_num), model_history.history["val_categorical_accuracy"], label="val_acc")
    except AttributeError:
        print("No prediction history available")
        return
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(file_path)
