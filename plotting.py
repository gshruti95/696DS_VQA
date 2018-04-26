import os
import sys
import matplotlib.pyplot as plt


SOURCE_FOLDER_PATH = '/home/srikanth/vqa/plotting_folder/'
DESTINATION_FOLDER_PATH = '/home/srikanth/vqa/plotting_images/'
def generate_loss_plots(plot_title, file_list, epoch_count):
    plt.clf()
    plt.title(plot_title)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    #plt.xlim((0, 100))
    #plt.ylim(0, 1.0)
    for file_name in file_list:
        temp_file = open(SOURCE_FOLDER_PATH + file_name)
        train_loss_list = []
        validation_loss_list = []
        count = 0
        for line in temp_file.readlines():
            # extract all the train and validation loss values
            word_list = line.replace('\n', '').split(' ')
            if word_list[0] == 'Loss':
                if count % 2 == 0:
                    train_loss_list.append(float(word_list[2]))
                else:
                    validation_loss_list.append((float(word_list[2])))
                count += 1

        temp_file.close()
        train_loss_list = train_loss_list[0 : epoch_count]
        validation_loss_list = validation_loss_list[0 : epoch_count]
        plt.plot(xrange(len(train_loss_list)), train_loss_list, label = 'Train loss')
        plt.plot(xrange(len(validation_loss_list)), validation_loss_list, label = 'Validation loss')
    
    plt.legend(loc = 3)
    #plt.show()
    plt.savefig(DESTINATION_FOLDER_PATH+ plot_title + '_loss.png')

def generate_accuracy_plots(plot_title, file_list, inital_value):
    plt.clf()
    plt.title(plot_title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.xlim((0, 100))
    plt.ylim(0, 1.0)
    for file_name in file_list:
        temp_file = open(SOURCE_FOLDER_PATH + file_name)
        train_accuracy_list = [inital_value]
        validation_accuracy_list = [inital_value]
        count = 0
        for line in temp_file.readlines():
            # extract all the train and validation loss values
            word_list = line.replace('\n', '').split(' ')
            if word_list[0] == 'Accuracy':
                if count % 2 == 0:
                    train_accuracy_list.append(float(word_list[2]))
                else:
                    validation_accuracy_list.append((float(word_list[2])))
                count += 1

        temp_file.close()
        plt.plot(xrange(len(train_accuracy_list)), train_accuracy_list, label = 'Train accuracy')
        plt.plot(xrange(len(validation_accuracy_list)), validation_accuracy_list, label = 'Validation accuracy')
    
    plt.legend(loc = 3)
    #plt.show()
    plt.savefig(DESTINATION_FOLDER_PATH + plot_title + '_accuracy.png' )


def main():
    file_names = os.listdir(SOURCE_FOLDER_PATH)
    for filename in file_names:
        print('plotting for file: ' + filename)
        plot_title = filename.replace('.txt', '')
        file_list = [filename]
        generate_accuracy_plots(plot_title + ' accuracy plot', file_list, 0.1)
        generate_loss_plots(plot_title + ' loss plot', file_list, 50)



if __name__ == '__main__':
    main()