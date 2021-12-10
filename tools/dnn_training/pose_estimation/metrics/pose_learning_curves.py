import matplotlib.pyplot as plt


class PoseLearningCurves:
    def __init__(self):
        self._training_loss_values = []
        self._validation_loss_values = []
        self._training_accuracy_values = []
        self._validation_accuracy_values = []
        self._training_map_values = []
        self._validation_map_values = []

    def clear(self):
        self._training_loss_values = []
        self._validation_loss_values = []
        self._training_accuracy_values = []
        self._validation_accuracy_values = []
        self._training_map_values = []
        self._validation_map_values = []

    def add_training_loss_value(self, value):
        self._training_loss_values.append(value)

    def add_validation_loss_value(self, value):
        self._validation_loss_values.append(value)

    def add_training_accuracy_value(self, value):
        self._training_accuracy_values.append(value)

    def add_validation_accuracy_value(self, value):
        self._validation_accuracy_values.append(value)

    def add_training_map_value(self, value):
        self._training_map_values.append(value)

    def add_validation_map_value(self, value):
        self._validation_map_values.append(value)

    def save_figure(self, output_path):
        fig = plt.figure(figsize=(15, 5), dpi=300)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        epochs = range(1, len(self._training_accuracy_values) + 1)
        ax1.plot(epochs, self._training_loss_values, '-o', color='tab:blue', label='Training')
        ax1.plot(epochs, self._validation_loss_values, '-o', color='tab:orange', label='Validation')
        ax1.set_title(u'Loss')
        ax1.set_xlabel(u'Epoch')
        ax1.set_ylabel(u'Loss')
        ax1.legend()

        epochs = range(1, len(self._training_accuracy_values) + 1)
        ax2.plot(epochs, self._training_accuracy_values, '-o', color='tab:blue', label='Training')
        ax2.plot(epochs, self._validation_accuracy_values, '-o', color='tab:orange', label='Validation')
        ax2.set_title(u'Accuracy')
        ax2.set_xlabel(u'Epoch')
        ax2.set_ylabel(u'Accuracy')
        ax2.legend()

        epochs = range(1, len(self._training_map_values) + 1)
        ax3.plot(epochs, self._training_map_values, '-o', color='tab:blue', label='Training')
        ax3.plot(epochs, self._validation_map_values, '-o', color='tab:orange', label='Validation')
        ax3.set_title(u'mAP')
        ax3.set_xlabel(u'Epoch')
        ax3.set_ylabel(u'mAP')
        ax3.legend()

        fig.savefig(output_path)
        plt.close(fig)
