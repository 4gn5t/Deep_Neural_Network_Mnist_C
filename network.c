#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  
#include <time.h>

#define INPUT_NODES 784  
#define HIDDEN_NODES 256 
#define OUTPUT_NODES 10  

#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000

#define NUMBER_OF_EPOCHS 10

double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

void load_mnist()
{
    FILE *training_images_file = fopen("mnist_train_images.bin", "rb");
    if (training_images_file == NULL)
    {
        printf("Error opening training images file\n");
        exit(1);
    }

    FILE *training_labels_file = fopen("mnist_train_labels.bin", "rb");
    if (training_labels_file == NULL)
    {
        printf("Error opening training labels file\n");
        exit(1);
    }

    FILE *test_images_file = fopen("mnist_test_images.bin", "rb");
    if (test_images_file == NULL)
    {
        printf("Error opening test images file\n");
        exit(1);
    }

    FILE *test_labels_file = fopen("mnist_test_labels.bin", "rb");
    if (test_labels_file == NULL)
    {
        printf("Error opening test labels file\n");
        exit(1);
    }

    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }

    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                training_labels[i][j] = 1;
            }
            else
            {
                training_labels[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }

    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, test_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                test_labels[i][j] = 1;
            }
            else
            {
                test_labels[i][j] = 0;
            }
        }
    }

    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

int max_index(double arr[], int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

void feedforward(double input[INPUT_NODES], 
                double weight1[INPUT_NODES][HIDDEN_NODES],
                double weight2[HIDDEN_NODES][OUTPUT_NODES],
                double bias1[HIDDEN_NODES],
                double bias2[OUTPUT_NODES],
                double hidden[HIDDEN_NODES],
                double output_layer[OUTPUT_NODES]) 
{
    
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j][i];
        }
        sum += bias1[i];
        hidden[i] = sigmoid(sum);
    }
    
    
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[j] * weight2[j][i];
        }
        sum += bias2[i];
        output_layer[i] = sigmoid(sum);
    }
}

void train(double input[INPUT_NODES], double output[OUTPUT_NODES], 
    double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], 
    double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], 
    int correct_label, int *correct_counter)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    
    feedforward(input, weight1, weight2, bias1, bias2, hidden, output_layer);

    int index = max_index(output_layer, OUTPUT_NODES);
    
    if (index == correct_label) {
        (*correct_counter)++;
    }
    
    
    double delta_output[OUTPUT_NODES];
    
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        delta_output[i] = (output[i] - output_layer[i]) * output_layer[i] * (1 - output_layer[i]);
    }
    
    double delta_hidden[HIDDEN_NODES];
    
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        delta_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            delta_hidden[i] += delta_output[j] * weight2[i][j];
        }
        delta_hidden[i] *= hidden[i] * (1 - hidden[i]);
    }
    
    
    double learning_rate = 0.1;
    
    
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] += learning_rate * delta_output[j] * hidden[i];
        }
    }
    
    
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] += learning_rate * delta_hidden[j] * input[i];
        }
    }
    
    
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] += learning_rate * delta_hidden[i];
    }
    
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] += learning_rate * delta_output[i];
    }
}

double init_nodes_weight()
{
    
    double xavier_init_hidden = sqrt(6.0 / (INPUT_NODES + HIDDEN_NODES));
    double xavier_init_output = sqrt(6.0 / (HIDDEN_NODES + OUTPUT_NODES));
    
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier_init_hidden;
        }
    }
    
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] = 0.0; 
        
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier_init_output;
        }
    }
    
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] = 0.0; 
    }
}

void test(double input[INPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES], 
    double weight2[HIDDEN_NODES][OUTPUT_NODES], double bias1[HIDDEN_NODES], 
    double bias2[OUTPUT_NODES], int correct_label, int *correct_counter)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    
    feedforward(input, weight1, weight2, bias1, bias2, hidden, output_layer);
    
    int index = max_index(output_layer, OUTPUT_NODES);
    
    if (index == correct_label) {
        (*correct_counter)++;
    }
}

void save_weights_biases(const char* filename)
{
    FILE* file = fopen(filename, "wb");
    if (file == NULL)
    {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    
    fwrite(weight1, sizeof(double), INPUT_NODES * HIDDEN_NODES, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
    fwrite(bias2, sizeof(double), OUTPUT_NODES, file);
    
    fclose(file);
    printf("Weights and biases saved to %s\n", filename);
}

int main()
{
    int correct_predictions = 0;
    
    
    srand(42); 
    
    
    clock_t start_time = clock();
    
    init_nodes_weight();
    load_mnist();

    printf("Starting training...\n");
    
    for(int epoch=0; epoch<NUMBER_OF_EPOCHS; epoch++)
    {
        int correct_train = 0;
        
        for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train(training_images[i], training_labels[i], weight1, weight2, bias1, bias2, correct_label, &correct_train);
            
            
            if ((i+1) % 10000 == 0) {
                printf("Progress: %d/%d images processed in epoch %d\n", i+1, NUM_TRAINING_IMAGES, epoch);
            }
        }
        printf("Epoch %d : Training Accuracy: %f\n", epoch, (double)correct_train / NUM_TRAINING_IMAGES);
    }
    save_weights_biases("model.bin");
    
    clock_t train_end_time = clock();
    printf("Training completed in %.2f seconds\n", (double)(train_end_time - start_time) / CLOCKS_PER_SEC);

    
    char choice;
    printf("\nDo you want to test specific images? (y/n): ");
    scanf(" %c", &choice);
    
    if (choice == 'y' || choice == 'Y') {
        while (1) {
            int image_index;
            printf("Enter image index (0-%d) or -1 to exit: ", NUM_TEST_IMAGES-1);
            scanf("%d", &image_index);
            
            if (image_index == -1) {
                break;
            }
            
            if (image_index < 0 || image_index >= NUM_TEST_IMAGES) {
                printf("Invalid image index. Please try again.\n");
                continue;
            }
            
            
            
            double hidden[HIDDEN_NODES];
            double output_layer[OUTPUT_NODES];
            
            
            feedforward(test_images[image_index], weight1, weight2, bias1, bias2, hidden, output_layer);
            
            int prediction = max_index(output_layer, OUTPUT_NODES);
            int true_label = max_index(test_labels[image_index], OUTPUT_NODES);
            
            printf("Prediction: %d\n", prediction);
            printf("True label: %d\n", true_label);
            printf("Result: %s\n\n", prediction == true_label ? "CORRECT" : "INCORRECT");
            
            printf("Confidence scores:\n");
            for (int i = 0; i < OUTPUT_NODES; i++) {
                printf("Digit %d: %.4f\n", i, output_layer[i]);
            }
            printf("\n");
        }
    }

    
    correct_predictions = 0;
    
    
    printf("Testing the network on %d images...\n", NUM_TEST_IMAGES);
    
    clock_t test_start_time = clock();
    
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], weight1, weight2, bias1, bias2, correct_label, &correct_predictions);
        
        
        if ((i+1) % 1000 == 0) {
            printf("Tested %d/%d images. Current accuracy: %f\n", 
                   i+1, NUM_TEST_IMAGES, (double)correct_predictions / (i+1));
        }
    }
    
    clock_t test_end_time = clock();
    printf("Testing completed in %.2f seconds\n", (double)(test_end_time - test_start_time) / CLOCKS_PER_SEC);
    printf("Total execution time: %.2f seconds\n", (double)(test_end_time - start_time) / CLOCKS_PER_SEC);
    
    printf("Testing Accuracy: %f\n", (double) correct_predictions / NUM_TEST_IMAGES);
    printf("Correct predictions: %d\n", correct_predictions);
    printf("Total test images: %d\n", NUM_TEST_IMAGES);

    return 0;
}
