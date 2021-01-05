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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.NASNet;
import org.deeplearning4j.zoo.model.ResNet50;
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

    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS; // acceptable image formats
    private static final long seed = 777; // seed used for randomizing data

    private static final int height = 200;
    private static final int width = 200;
    private static final int channels = 3;

    private static final Random randNumGen = new Random(seed);

    private final String baseDir = System.getProperty("user.dir") + "\\OpenIMAJ-CW3\\";

    private final int batchSize = 10;
    private static final int classes = 15;

    private static final int epochs = 10;


    private static DataSetIterator trainIter;
    private static DataSetIterator testIter;

    private ZooModel nASNet;
    private ZooModel rESNet50;

    private int lastLayerIndex;


    public CNN() throws IOException, InterruptedException {


        setUpTrainTest();
//        rESNet50 = resNETFromScratch(); // run this one please
        nASNet = NABSNetPredefined();

    }

    public ZooModel NABSNetPredefined() throws IOException {


        ZooModel nASNet = NASNet.builder().build();
        ComputationGraph net = (ComputationGraph) nASNet.initPretrained(PretrainedType.IMAGENETLARGE);

        lastLayerIndex = net.getLayers().length - 1;


        Layer lastLayer = net.getLayers()[lastLayerIndex];
        String lastLayerName = lastLayer.conf().getLayer().getLayerName();

        Layer featureLayer = net.getLayers()[lastLayerIndex - 1];
        String featureLayerName = featureLayer.conf().getLayer().getLayerName();

        FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();


        ComputationGraph nasNetTransfer = new TransferLearning.GraphBuilder(net)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor(featureLayerName)
                .addLayer("connector",
                        new DenseLayer.Builder()
                                .nIn(1000)
                                .nOut(256)
                                .activation(Activation.RELU)
                                .build(),
                        lastLayerName)
                .addLayer("new_predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .nIn(256)
                                .nOut(classes)
                                .build(),
                        "connector")
                .setOutputs("new_predictions")
                .build();


        nasNetTransfer.setListeners(
                new ScoreIterationListener(1),
                new PerformanceListener(1),
                new TimeIterationListener(1));

        double lowest = 10;

        System.out.println("\n --- Training Network ---");

        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            nasNetTransfer.fit(trainIter);
            System.out.println("epoch: " + i + " score: " + nasNetTransfer.score());

            if (nasNetTransfer.score() < lowest) {
                lowest = nasNetTransfer.score();
                System.out.println("epoch: " + i + " lowest: " + lowest);
                String modelFilename = String.format("%s_lowest_err_%f_epoch_%d_NASNet.zip", baseDir, lowest, i);
                ModelSerializer.writeModel(nasNetTransfer, modelFilename, false);
            }
        }

        nasNetTransfer.setListeners(
                new EvaluativeListener(trainIter, 1),
                new EvaluativeListener(testIter, 1));

        ModelSerializer.writeModel(nasNetTransfer, "Final_NASNet.zip", false);

        trainIter.reset();
        Evaluation trainEval = nasNetTransfer.evaluate(trainIter);
        Evaluation testEval = nasNetTransfer.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());

        return nASNet;
    }

    public ZooModel resNETFromScratch() throws IOException {


        ZooModel resNet50 = ResNet50.builder()
                .numClasses(classes)
                .seed(seed)
                .inputShape(new int[]{channels, height, width})
                .updater(new Adam(1e-3))
                .build();

        ComputationGraph net = resNet50.init();


        net.setListeners(
                new ScoreIterationListener(1),
                new PerformanceListener(1),
                new TimeIterationListener(1));

        System.out.println("\n --- Training Network ---");

        double lowest = 10;

        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            net.fit(trainIter);
            System.out.println("epoch: " + i + " score: " + net.score());

            if (net.score() < lowest) {
                lowest = net.score();
                System.out.println("epoch: " + i + " lowest: " + lowest);
                String modelFilename = String.format("%s_lowest_err_%f_epoch_%d_RESNet50.zip", baseDir, lowest, i);
                ModelSerializer.writeModel(net, modelFilename, false);
            }
        }

        net.setListeners(
                new EvaluativeListener(trainIter, 1),
                new EvaluativeListener(testIter, 1)
        );

        ModelSerializer.writeModel(net, "Final_RESNet50.zip", false);

        trainIter.reset();
        Evaluation trainEval = net.evaluate(trainIter);
        Evaluation testEval = net.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());

        return resNet50;
    }

    private void setUpTrainTest() throws IOException, InterruptedException {

        File directory = new File(baseDir, "training/");
        FileSplit fileSplit = new FileSplit(directory, allowedExtensions, randNumGen);

        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelGenerator);

        InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelGenerator);
        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelGenerator);

        // Can add more transforms to augment the data
        ImageTransform transform = new MultiImageTransform(randNumGen);

        trainRecordReader.initialize(trainData, transform);
        testRecordReader.initialize(testData, transform);

        trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, classes);
        testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, classes);

        ImagePreProcessingScaler normalizer = new ImagePreProcessingScaler();

        trainIter.setPreProcessor(normalizer);
        testIter.setPreProcessor(normalizer);

    }

    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
