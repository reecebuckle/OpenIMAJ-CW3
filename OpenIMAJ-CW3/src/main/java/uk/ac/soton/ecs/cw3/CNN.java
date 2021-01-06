package uk.ac.soton.ecs.cw3;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.*;

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;


public class CNN {
    /**
     * CNN Class used to create train and test a simple CNN
     */

    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS; // acceptable image formats
    private static final long seed = 777; // seed used for randomizing data

    private static final int height = 256; // image height to crop to
    private static final int width = 256; // image width to crop to
    private static final int channels = 1; // number of channels set to 1 for greyscale

    private static final Random randNumGen = new Random(seed);

    private final String baseDir = System.getProperty("user.dir") + "\\OpenIMAJ-CW3\\";

    private final int batchSize = 10; // the batch size for training
    private static final int classes = 15; // the number of classes to classify

    private static final int epochs = 50; // the number of epochs to train data with


    private static DataSetIterator trainIter; // the train set
    private static DataSetIterator testIter; // the test set

    private ZooModel rESNet50; // preconfigured RESNet50


    public CNN () throws IOException, InterruptedException {
        /**
         * Constructor sets up the training data and a simple CNN.
         */

        setUpTrainTest(); // sets up the training and testing data
//        resNETPredefined(); // initialises and runs an augmented resNet50 using transfer learning
        simpleCNN(); // initialises and runs a cin


    }


    public void simpleCNN() throws IOException {
        /**
         * Method creates a simple CNN, then trains and tests it
         */

        // create a pre-made simple CNN to train
        ZooModel model = SimpleCNN.builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                .inputShape(new int[] {channels, height, width})
                .cacheMode(CacheMode.DEVICE)
                .seed(seed)
                .numClasses(classes)
                .updater(new Adam(1e-2, 0.001, 0.999, 0.1))
                .build();


        // Initialise the cnn to create a Multi layered neural network
        MultiLayerNetwork computationGraph =  model.init();

        // create the fine tune configuration for tuning the CNN
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(1e-2))
                .backpropType(BackpropType.Standard)
                .seed(seed)
                .build();

        // Add an output layer to the CNN for classification
        MultiLayerNetwork net = new TransferLearning.Builder(computationGraph)
                .fineTuneConfiguration(fineTuneConf)
                .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(15)
                        .nOut(15)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // Set listeners fir evaluating metrics during training and estimate training time
        net.setListeners(
                new ScoreIterationListener(1),
                new PerformanceListener(1),
                new TimeIterationListener((epochs*classes*100)/batchSize));

        System.out.println("\n --- Training Network ---");


        MultiLayerConfiguration high; // Used to determine the lowest scoring network at a given epoch
        String modelFilename = null; // filename for the lowest scoring network at a given epoch
        double lowest = Double.MAX_VALUE; // the value of the score at a given epoch

        MultiLayerNetwork trained = null; // the trained network with the lowest score at a given epoch

        for (int i = 0; i < epochs; i++) {
            trainIter.reset(); // resets the training set at each epoch
            net.fit(trainIter); // fits the training set to the network
            System.err.println("epoch: " + i + " score: " + net.score());

            // determines if the current trained network is has the lowest score
            if (net.score() < lowest) {
                lowest = net.score();
                System.out.println("epoch: " + i + " highest: " + lowest);
                modelFilename = String.format("%s_lowest_%f_epoch_%d_convolution.zip", baseDir, lowest, i);
                net.save(new File(modelFilename));
                trained = net.clone();
            }
        }

        // sets evaluation listeners
        trained.setListeners(
                new EvaluativeListener(trainIter, 1),
                new EvaluativeListener(testIter, 1)
        );

        // evaluates the training and testing data on the network
        trainIter.reset();
        Evaluation trainEval = trained.evaluate(trainIter);
        Evaluation testEval = trained.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());
    }

    private void setUpTrainTest() throws IOException {
        /**
         * Method for seting up the training test split on the input data
         */
        // gets the child folders inside the training folder
        File directory = new File(baseDir, "training/");
        // splits the data randomly
        FileSplit fileSplit = new FileSplit(directory, allowedExtensions, randNumGen);

        // generates the class labels
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        //balances the splits to evenly distribute data from classes in each batch
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelGenerator);
        //plits the data into test and training sets with a 80:20 split
        InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        // Readers used to read the test and training data
        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelGenerator);
        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelGenerator);

        // Can add more transforms to augment the data
        ImageTransform transform = new MultiImageTransform(randNumGen);

        // gets the test and train data
        trainRecordReader.initialize(trainData, transform);
        testRecordReader.initialize(testData, transform);

        // Creates the test and training datasets
        trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, classes);
        testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, classes);

        // Normaliser used to normalise the data
        ImagePreProcessingScaler normalizer = new ImagePreProcessingScaler();

        // normalizes the data
        trainIter.setPreProcessor(normalizer);
        testIter.setPreProcessor(normalizer);

    }
}
