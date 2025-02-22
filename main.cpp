#include <Arduino.h>
#include <TensorFlowLite.h>
#include "sin_predictor_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <math.h>

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 7  

// Increased Tensor Arena to 50KB for better model execution
constexpr int tensor_arena_size = 50 * 1024;  
static uint8_t tensor_arena[tensor_arena_size];  

// Static TensorFlow Lite Model, Interpreter, and Tensors
static const tflite::Model* model = tflite::GetModel(sin_predictor_int8_tflite);
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, nullptr);

static TfLiteTensor* input_tensor;
static TfLiteTensor* output_tensor;

// Serial Communication Buffers
char received_char = (char)NULL;              
int chars_avail = 0;                    
char out_str_buff[OUTPUT_BUFFER_SIZE];  
char in_str_buff[INPUT_BUFFER_SIZE];    
int input_array[INT_ARRAY_SIZE];       
int in_buff_idx = 0; 
int array_length = 0;

// Function Declarations
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
void run_inference(int *int_array);
unsigned long measure_time_in_us();

void setup() {
    delay(5000);
    Serial.begin(115200);
    Serial.println("TFLM Sine Predictor Ready!");

    // Ensure Model Loads Correctly
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors failed!");
        while (1);
    }

    // Get references to input/output tensors
    input_tensor = interpreter.input(0);
    output_tensor = interpreter.output(0);

    memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
}

void loop() {
    chars_avail = Serial.available();
    if (chars_avail > 0) {
        received_char = Serial.read();
        Serial.print(received_char);  

        if (in_buff_idx < INPUT_BUFFER_SIZE - 1) {  
            in_str_buff[in_buff_idx++] = received_char;
        } else {  
            Serial.println("\nERROR: Input too long! Resetting buffer.");
            memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
            in_buff_idx = 0;
        }

        if (received_char == 13) {  // User pressed 'Enter'
            Serial.println("\nProcessing input...");
            array_length = string_to_array(in_str_buff, input_array);

            if (array_length == INT_ARRAY_SIZE) {
                Serial.println("Valid input. Running inference...");
                print_int_array(input_array, array_length);

                // Measure execution time
                unsigned long t0 = measure_time_in_us();
                Serial.println("test statement");
                unsigned long t1 = measure_time_in_us();

                run_inference(input_array);
                unsigned long t2 = measure_time_in_us();

                unsigned long t_print = t1 - t0;
                unsigned long t_infer = t2 - t1;

                Serial.print("Printing time = ");
                Serial.print(t_print);
                Serial.print(" us.  Inference time = ");
                Serial.print(t_infer);
                Serial.println(" us.");
            } else {
                Serial.println("ERROR: Please enter exactly 7 numbers.");
            }

            memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
            in_buff_idx = 0;
        }
    }
}

// Converts input string to integer array
int string_to_array(char *in_str, int *int_array) {
    int num_integers = 0;
    char *token = strtok(in_str, ",");
  
    while (token != NULL) {
        int_array[num_integers++] = atoi(token);
        token = strtok(NULL, ",");
        if (num_integers >= INT_ARRAY_SIZE) break;
    }
    return num_integers;
}

// Print input numbers
void print_int_array(int *int_array, int array_len) {
    Serial.print("Input: [");
    for (int i = 0; i < array_len; i++) {
        Serial.print(int_array[i]);
        if (i < array_len - 1) Serial.print(", ");
    }
    Serial.println("]");
}

// Updated Inference Function to Predict Both Number and Sine Value
void run_inference(int *int_array) {
    Serial.println("Quantizing input values...");
    
    // Normalize input values correctly
    for (int i = 0; i < INT_ARRAY_SIZE; i++) {
        float normalized_input = (float)int_array[i] / 10.0f;  
        input_tensor->data.int8[i] = (int8_t)(normalized_input * 127.0f);

        Serial.print("Quantized Input ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(input_tensor->data.int8[i]);
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        Serial.println("ERROR: Inference failed!");
        return;
    }

    // Retrieve quantized prediction
    int8_t predicted_value = output_tensor->data.int8[0];

    // Properly dequantize output to match the expected range [-1,1]
    float predicted_number = (float)predicted_value / 127.0f;

    // Compute sine wave prediction
    float predicted_sine = sin(predicted_number * M_PI);  

    // Print results
    Serial.print("Predicted next number: ");
    Serial.println(predicted_number, 4);
    
    Serial.print("Predicted sine wave value: ");
    Serial.println(predicted_sine, 4);
}

// Measure execution time in microseconds
unsigned long measure_time_in_us() {
    return micros();
}
