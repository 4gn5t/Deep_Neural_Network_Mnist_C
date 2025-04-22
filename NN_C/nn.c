#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  
#include <time.h>

#define INPUT_NODES 784
#define HIDDEN_NODES_LAYER1 256
#define HIDDEN_NODES_LAYER2 128
#define HIDDEN_NODES_LAYER3 64 
#define OUTPUT_NODES 10
#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define NUMBER_OF_EPOCHS 10
#define BATCH_SIZE 32

double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

double weight1[INPUT_NODES][HIDDEN_NODES_LAYER1];
double weight2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2];
double weight3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3];
double weight4[HIDDEN_NODES_LAYER3][OUTPUT_NODES];
double bias1[HIDDEN_NODES_LAYER1];
double bias2[HIDDEN_NODES_LAYER2];
double bias3[HIDDEN_NODES_LAYER3];
double bias4[OUTPUT_NODES];

void load_mnist()
{
    FILE *training_images_file = fopen("../MNIST_bin/mnist_train_images.bin", "rb");
    if (training_images_file == NULL)
    {
        printf("Error opening training images file\n");
        exit(1);
    }

    FILE *training_labels_file = fopen("../MNIST_bin/mnist_train_labels.bin", "rb");
    if (training_labels_file == NULL)
    {
        printf("Error opening training labels file\n");
        exit(1);
    }

    FILE *test_images_file = fopen("../MNIST_bin/mnist_test_images.bin", "rb");
    if (test_images_file == NULL)
    {
        printf("Error opening test images file\n");
        exit(1);
    }

    FILE *test_labels_file = fopen("../MNIST_bin/mnist_test_labels.bin", "rb");
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

int max_index(double arr[], int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

void save_weights_biases(const char* filename)
{
    FILE* file = fopen(filename, "wb");
    if (file == NULL)
    {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save all weights and biases
    fwrite(weight1, sizeof(double), INPUT_NODES * HIDDEN_NODES_LAYER1, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES_LAYER1 * HIDDEN_NODES_LAYER2, file);
    fwrite(weight3, sizeof(double), HIDDEN_NODES_LAYER2 * HIDDEN_NODES_LAYER3, file);
    fwrite(weight4, sizeof(double), HIDDEN_NODES_LAYER3 * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES_LAYER1, file);
    fwrite(bias2, sizeof(double), HIDDEN_NODES_LAYER2, file);
    fwrite(bias3, sizeof(double), HIDDEN_NODES_LAYER3, file);
    fwrite(bias4, sizeof(double), OUTPUT_NODES, file);
    
    fclose(file);
    printf("Weights and biases saved to %s\n", filename);
}


double relu(double x)
{
    return (x > 0) ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

void feedforward(
    double input[INPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES_LAYER1],
    double weight2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double weight3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3],
    double weight4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double bias1[HIDDEN_NODES_LAYER1],
    double bias2[HIDDEN_NODES_LAYER2],
    double bias3[HIDDEN_NODES_LAYER3],
    double bias4[OUTPUT_NODES],
    double hidden1[HIDDEN_NODES_LAYER1],
    double hidden2[HIDDEN_NODES_LAYER2],
    double hidden3[HIDDEN_NODES_LAYER3],
    double output_layer[OUTPUT_NODES]
) {
    // Layer 1
    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++) {
        double sum = 0;
        for (int j = 0; j < INPUT_NODES; j++) {
            sum += input[j] * weight1[j][i];
        }
        sum += bias1[i];
        hidden1[i] = relu(sum);
    }
    // Layer 2
    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES_LAYER1; j++) {
            sum += hidden1[j] * weight2[j][i];
        }
        sum += bias2[i];
        hidden2[i] = relu(sum);
    }
    // Layer 3
    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES_LAYER2; j++) {
            sum += hidden2[j] * weight3[j][i];
        }
        sum += bias3[i];
        hidden3[i] = relu(sum);
    }
    // Output
    for (int i = 0; i < OUTPUT_NODES; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES_LAYER3; j++) {
            sum += hidden3[j] * weight4[j][i];
        }
        sum += bias4[i];
        output_layer[i] = relu(sum);
    }
}

void backpropagation(
    double input[INPUT_NODES], double target[OUTPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES_LAYER1], double weight2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double weight3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], double weight4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double bias1[HIDDEN_NODES_LAYER1], double bias2[HIDDEN_NODES_LAYER2], double bias3[HIDDEN_NODES_LAYER3], double bias4[OUTPUT_NODES],
    double hidden1[HIDDEN_NODES_LAYER1], double hidden2[HIDDEN_NODES_LAYER2], double hidden3[HIDDEN_NODES_LAYER3], double output_layer[OUTPUT_NODES],
    double delta1_hidden[HIDDEN_NODES_LAYER1], double delta2_hidden[HIDDEN_NODES_LAYER2], double delta3_hidden[HIDDEN_NODES_LAYER3], double delta_output[OUTPUT_NODES]
) {
    // Output layer delta
    for (int i = 0; i < OUTPUT_NODES; i++) {
        delta_output[i] = (output_layer[i] - target[i]) * relu_derivative(output_layer[i]);
    }
    // Hidden layer 3 delta
    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++) {
        double sum = 0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            sum += delta_output[j] * weight4[i][j];
        }
        delta3_hidden[i] = sum * relu_derivative(hidden3[i]);
    }
    // Hidden layer 2 delta
    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES_LAYER3; j++) {
            sum += delta3_hidden[j] * weight3[i][j];
        }
        delta2_hidden[i] = sum * relu_derivative(hidden2[i]);
    }
    // Hidden layer 1 delta
    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++) {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES_LAYER2; j++) {
            sum += delta2_hidden[j] * weight2[i][j];
        }
        delta1_hidden[i] = sum * relu_derivative(hidden1[i]);
    }
}

void zero_gradients(
    double dW1[INPUT_NODES][HIDDEN_NODES_LAYER1], double dW2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double dW3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], double dW4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double db1[HIDDEN_NODES_LAYER1], double db2[HIDDEN_NODES_LAYER2], double db3[HIDDEN_NODES_LAYER3], double db4[OUTPUT_NODES]
) {
    memset(dW1, 0, sizeof(double) * INPUT_NODES * HIDDEN_NODES_LAYER1);
    memset(dW2, 0, sizeof(double) * HIDDEN_NODES_LAYER1 * HIDDEN_NODES_LAYER2);
    memset(dW3, 0, sizeof(double) * HIDDEN_NODES_LAYER2 * HIDDEN_NODES_LAYER3);
    memset(dW4, 0, sizeof(double) * HIDDEN_NODES_LAYER3 * OUTPUT_NODES);
    memset(db1, 0, sizeof(double) * HIDDEN_NODES_LAYER1);
    memset(db2, 0, sizeof(double) * HIDDEN_NODES_LAYER2);
    memset(db3, 0, sizeof(double) * HIDDEN_NODES_LAYER3);
    memset(db4, 0, sizeof(double) * OUTPUT_NODES);
}

void update_weights(
    double weight1[INPUT_NODES][HIDDEN_NODES_LAYER1], double weight2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double weight3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], double weight4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double bias1[HIDDEN_NODES_LAYER1], double bias2[HIDDEN_NODES_LAYER2], double bias3[HIDDEN_NODES_LAYER3], double bias4[OUTPUT_NODES],
    double dW1[INPUT_NODES][HIDDEN_NODES_LAYER1], double dW2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double dW3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], double dW4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double db1[HIDDEN_NODES_LAYER1], double db2[HIDDEN_NODES_LAYER2], double db3[HIDDEN_NODES_LAYER3], double db4[OUTPUT_NODES],
    double learning_rate, int batch_size
) {
    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER1; j++)
            weight1[i][j] -= learning_rate * dW1[i][j] / batch_size;
    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER2; j++)
            weight2[i][j] -= learning_rate * dW2[i][j] / batch_size;
    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER3; j++)
            weight3[i][j] -= learning_rate * dW3[i][j] / batch_size;
    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++)
        for (int j = 0; j < OUTPUT_NODES; j++)
            weight4[i][j] -= learning_rate * dW4[i][j] / batch_size;
    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++) bias1[i] -= learning_rate * db1[i] / batch_size;
    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++) bias2[i] -= learning_rate * db2[i] / batch_size;
    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++) bias3[i] -= learning_rate * db3[i] / batch_size;
    for (int i = 0; i < OUTPUT_NODES; i++) bias4[i] -= learning_rate * db4[i] / batch_size;
}

void accumulate_gradients(
    double input[INPUT_NODES], double target[OUTPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES_LAYER1], double weight2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double weight3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], double weight4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double bias1[HIDDEN_NODES_LAYER1], double bias2[HIDDEN_NODES_LAYER2], double bias3[HIDDEN_NODES_LAYER3], double bias4[OUTPUT_NODES],
    double dW1[INPUT_NODES][HIDDEN_NODES_LAYER1], double dW2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double dW3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], double dW4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double db1[HIDDEN_NODES_LAYER1], double db2[HIDDEN_NODES_LAYER2], double db3[HIDDEN_NODES_LAYER3], double db4[OUTPUT_NODES],
    int correct_label, int *correct_counter
) {
    double hidden1[HIDDEN_NODES_LAYER1];
    double hidden2[HIDDEN_NODES_LAYER2];
    double hidden3[HIDDEN_NODES_LAYER3];
    double output_layer[OUTPUT_NODES];

    feedforward(input, weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4, hidden1, hidden2, hidden3, output_layer);

    int index = max_index(output_layer, OUTPUT_NODES);
    if (index == correct_label) {
        (*correct_counter)++;
    }

    double delta_output[OUTPUT_NODES];
    double delta3_hidden[HIDDEN_NODES_LAYER3];
    double delta2_hidden[HIDDEN_NODES_LAYER2];
    double delta1_hidden[HIDDEN_NODES_LAYER1];

    backpropagation(input, target, weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4,
                   hidden1, hidden2, hidden3, output_layer,
                   delta1_hidden, delta2_hidden, delta3_hidden, delta_output);

    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++)
        for (int j = 0; j < OUTPUT_NODES; j++)
            dW4[i][j] += delta_output[j] * hidden3[i];
    for (int i = 0; i < OUTPUT_NODES; i++)
        db4[i] += delta_output[i];

    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER3; j++)
            dW3[i][j] += delta3_hidden[j] * hidden2[i];
    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++)
        db3[i] += delta3_hidden[i];

    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER2; j++)
            dW2[i][j] += delta2_hidden[j] * hidden1[i];
    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++)
        db2[i] += delta2_hidden[i];

    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER1; j++)
            dW1[i][j] += delta1_hidden[j] * input[i];
    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++)
        db1[i] += delta1_hidden[i];
}

void train(
    double input[INPUT_NODES], double target[OUTPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES_LAYER1], double weight2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double weight3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], double weight4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double bias1[HIDDEN_NODES_LAYER1], double bias2[HIDDEN_NODES_LAYER2], double bias3[HIDDEN_NODES_LAYER3], double bias4[OUTPUT_NODES],
    int correct_label, int *correct_counter
) {
    double hidden1[HIDDEN_NODES_LAYER1];
    double hidden2[HIDDEN_NODES_LAYER2];
    double hidden3[HIDDEN_NODES_LAYER3];
    double output_layer[OUTPUT_NODES];

    feedforward(input, weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4, hidden1, hidden2, hidden3, output_layer);

    int index = max_index(output_layer, OUTPUT_NODES);
    if (index == correct_label) {
        (*correct_counter)++;
    }

    double delta_output[OUTPUT_NODES];
    double delta3_hidden[HIDDEN_NODES_LAYER3];
    double delta2_hidden[HIDDEN_NODES_LAYER2];
    double delta1_hidden[HIDDEN_NODES_LAYER1];

    backpropagation(input, target, weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4,
                   hidden1, hidden2, hidden3, output_layer,
                   delta1_hidden, delta2_hidden, delta3_hidden, delta_output);
}

void init_nodes_weight()
{
    double xavier1 = sqrt(6.0 / (INPUT_NODES + HIDDEN_NODES_LAYER1));
    double xavier2 = sqrt(6.0 / (HIDDEN_NODES_LAYER1 + HIDDEN_NODES_LAYER2));
    double xavier3 = sqrt(6.0 / (HIDDEN_NODES_LAYER2 + HIDDEN_NODES_LAYER3));
    double xavier4 = sqrt(6.0 / (HIDDEN_NODES_LAYER3 + OUTPUT_NODES));

    for (int i = 0; i < INPUT_NODES; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER1; j++)
            weight1[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier1;

    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER2; j++)
            weight2[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier2;

    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++)
        for (int j = 0; j < HIDDEN_NODES_LAYER3; j++)
            weight3[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier3;

    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++)
        for (int j = 0; j < OUTPUT_NODES; j++)
            weight4[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * xavier4;

    for (int i = 0; i < HIDDEN_NODES_LAYER1; i++) bias1[i] = 0.0;
    for (int i = 0; i < HIDDEN_NODES_LAYER2; i++) bias2[i] = 0.0;
    for (int i = 0; i < HIDDEN_NODES_LAYER3; i++) bias3[i] = 0.0;
    for (int i = 0; i < OUTPUT_NODES; i++) bias4[i] = 0.0;
}

void test(
    double input[INPUT_NODES],
    double weight1[INPUT_NODES][HIDDEN_NODES_LAYER1],
    double weight2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2],
    double weight3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3],
    double weight4[HIDDEN_NODES_LAYER3][OUTPUT_NODES],
    double bias1[HIDDEN_NODES_LAYER1], double bias2[HIDDEN_NODES_LAYER2], double bias3[HIDDEN_NODES_LAYER3], double bias4[OUTPUT_NODES],
    int correct_label, int *correct_counter
) {
    double hidden1[HIDDEN_NODES_LAYER1];
    double hidden2[HIDDEN_NODES_LAYER2];
    double hidden3[HIDDEN_NODES_LAYER3];
    double output_layer[OUTPUT_NODES];

    feedforward(input, weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4, hidden1, hidden2, hidden3, output_layer);

    int index = max_index(output_layer, OUTPUT_NODES);
    if (index == correct_label) {
        (*correct_counter)++;
    }
}

int main()
{
    int correct_predictions = 0;
    srand(42); 
    clock_t start_time = clock();

    init_nodes_weight();
    load_mnist();

    printf("Starting training...\n");

    double learning_rate = 0.001;

    for(int epoch=0; epoch<NUMBER_OF_EPOCHS; epoch++)
    {
        int correct_train = 0;

        double dW1[INPUT_NODES][HIDDEN_NODES_LAYER1], dW2[HIDDEN_NODES_LAYER1][HIDDEN_NODES_LAYER2];
        double dW3[HIDDEN_NODES_LAYER2][HIDDEN_NODES_LAYER3], dW4[HIDDEN_NODES_LAYER3][OUTPUT_NODES];
        double db1[HIDDEN_NODES_LAYER1], db2[HIDDEN_NODES_LAYER2], db3[HIDDEN_NODES_LAYER3], db4[OUTPUT_NODES];

        for (int batch_start = 0; batch_start < NUM_TRAINING_IMAGES; batch_start += BATCH_SIZE)
        {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > NUM_TRAINING_IMAGES) batch_end = NUM_TRAINING_IMAGES;
            int batch_size = batch_end - batch_start;

            zero_gradients(dW1, dW2, dW3, dW4, db1, db2, db3, db4);

            for (int i = batch_start; i < batch_end; i++)
            {
                int correct_label = max_index(training_labels[i], OUTPUT_NODES);
                accumulate_gradients(training_images[i], training_labels[i], weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4,
                                    dW1, dW2, dW3, dW4, db1, db2, db3, db4, correct_label, &correct_train);
            }

            update_weights(weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4,
                           dW1, dW2, dW3, dW4, db1, db2, db3, db4, learning_rate, batch_size);

            if ((batch_end) % 10000 == 0) {
                printf("Progress: %d/%d images processed in epoch %d\n", batch_end, NUM_TRAINING_IMAGES, epoch);
            }
        }

        printf("Epoch %d | Training Accuracy: %f\n", epoch, (double)correct_train / NUM_TRAINING_IMAGES);
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
            
            double hidden1[HIDDEN_NODES_LAYER1];
            double hidden2[HIDDEN_NODES_LAYER2];
            double hidden3[HIDDEN_NODES_LAYER3];
            double output_layer[OUTPUT_NODES];

            feedforward(test_images[image_index], weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4, hidden1, hidden2, hidden3, output_layer);
            
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
        test(test_images[i], weight1, weight2, weight3, weight4, bias1, bias2, bias3, bias4, correct_label, &correct_predictions);
        
        
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
