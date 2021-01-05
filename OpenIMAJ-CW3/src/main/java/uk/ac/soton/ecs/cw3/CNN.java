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
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitXavier;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.TimeIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.NASNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
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
    

    public CNN () throws IOException, InterruptedException {


        setUpTrainTest();
//        rESNet50 = resNETFromScratch(); // run this one please
//        NASNetFromScratch(); // run this one
//        resNETPredefined();
        VGG16Predefined();
//        nASNet = NABSNetPredefined();


    }

    public void NASNetFromScratch() throws IOException {

        NASNet nasNET = NASNet.builder()
                .numClasses(classes)
                .seed(seed)
                .inputShape(new int[]{1, height, width})
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .penultimateFilters(120)
                .numBlocks(5)
                .filterMultiplier(3)
                .stemFilters(3)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                .skipReduction(false)
                .build();

        System.out.println(nasNET.metaData().getInputShape()[0].toString());
        System.out.println(nasNET.graphBuilder().getVertices());


//        ComputationGraph net = nasNET.init();
//        System.out.println(net.getLayers());
//
//        FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .updater(new Nesterovs(5e-5))
//                .seed(seed)
//                .build();
//
//
//        ComputationGraph nasNetTransfer = new TransferLearning.GraphBuilder(net)
//                .fineTuneConfiguration(fineTuneConfiguration)
//                .setInputs()
//                .build();
//
//        net.setListeners(
//                new ScoreIterationListener(1),
//                new PerformanceListener(1),
//                new TimeIterationListener((5*classes*100)/batchSize));
//
//        System.out.println("\n --- Training Network ---");
//
//        double lowest = 10;
//
//        for (int i = 0; i < 5; i++) {
//            trainIter.reset();
//            net.fit(trainIter);
//            System.out.println("epoch: " + i + " score: " + net.score());
//
//            if (net.score() < lowest) {
//                lowest = net.score();
//                System.out.println("epoch: " + i + " lowest: " + lowest);
//                String modelFilename = String.format("%s_lowest_err_%f_epoch_%d_RESNet50scratch.zip", baseDir, lowest, i);
//                ModelSerializer.writeModel(net, modelFilename, false);
//            }
//        }
//
//        net.setListeners(
//                new EvaluativeListener(trainIter, 1),
//                new EvaluativeListener(testIter, 1)
//        );
//
//        ModelSerializer.writeModel(net, "Final_RESNet50scratch.zip", false);
//
//        trainIter.reset();
//        Evaluation trainEval = net.evaluate(trainIter);
//        Evaluation testEval = net.evaluate(testIter);
//
//        System.out.println(trainEval.stats());
//        System.out.println(testEval.stats());

    }

    public ZooModel NABSNetPredefined() throws IOException {


        ZooModel nASNet = NASNet.builder().build();
        ComputationGraph net = (ComputationGraph) nASNet.initPretrained(PretrainedType.IMAGENETLARGE);

        lastLayerIndex = net.getLayers().length - 1;


        Layer lastLayer = net.getLayers()[lastLayerIndex];
        String lastLayerName = lastLayer.conf().getLayer().getLayerName();

        Layer featureLayer = net.getLayers()[lastLayerIndex-1];
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
                new TimeIterationListener((epochs*classes*100)/batchSize));

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
                .inputShape(new int[]{1, height, width})
                .updater(new Adam(1e-3))
                .weightInit(new WeightInitXavier())
                .build();

        ComputationGraph net = resNet50.init();


        net.setListeners(
                new ScoreIterationListener(1),
                new PerformanceListener(1),
                new TimeIterationListener((5*classes*100)/batchSize));

        System.out.println("\n --- Training Network ---");

        double lowest = 10;

        for (int i = 0; i < 5; i++) {
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

    public void VGG16Predefined() throws IOException {
        
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();

        ComputationGraph net = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096)
                                .nOut(classes)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();

        net.setListeners(
                new ScoreIterationListener(1),
                new PerformanceListener(1),
                new TimeIterationListener((5*classes*100)/batchSize));

        System.out.println("\n --- Training Network ---");

        MultiLayerConfiguration high;
        String modelFilename = null;
        double highest = Double.MIN_VALUE;

        ComputationGraph trained = null;

        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            net.fit(trainIter);
            System.out.println("epoch: " + i + " score: " + net.score());

            if (net.score() > highest) {
                highest = net.score();
                System.out.println("epoch: " + i + " highest: " + highest);
                modelFilename = String.format("%s_lowest_err_%f_epoch_%d_RESNet50_transfer.zip", baseDir, highest, i);
                net.save(new File(modelFilename));
                trained = net.clone();
            }
        }

        trained.setListeners(
                new EvaluativeListener(trainIter, 1),
                new EvaluativeListener(testIter, 1)
        );

        trainIter.reset();
        Evaluation trainEval = trained.evaluate(trainIter);
        Evaluation testEval = trained.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());
    }

    public void resNETPredefined() throws IOException {


        ZooModel resNet50 = ResNet50.builder().build();

        ComputationGraph net = (ComputationGraph) resNet50.initPretrained();

        int c = 0;
        for (Layer layer : net.getLayers()) {
            System.out.println(c + " " + layer);
            c ++;
        }

        String finalLayer = net.getLayers()[157].conf().getLayer().getLayerName();
        System.out.println(finalLayer);
        String featureLayer = net.getLayers()[154].conf().getLayer().getLayerName();
        System.out.println(featureLayer);
        String attachLayer = net.getLayers()[156].conf().getLayer().getLayerName();

        FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .updater(new Adam(1e-3))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(seed)
                .build();

        ComputationGraph resNet50Transfer = new TransferLearning.GraphBuilder(net)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor(featureLayer)
//                .nOutReplace(finalLayer, classes, WeightInit.XAVIER)
//                .setOutputs(finalLayer)
                .removeVertexKeepConnections(finalLayer)
//                .addLayer("fc-256", new DenseLayer.Builder()
//                    .activation(Activation.RELU)
//                    .nIn(1000)
//                    .nOut(256)
//                    .build(), featureLayer)
                .addLayer(finalLayer, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(2048)
                        .nOut(classes)
                        .activation(Activation.SOFTMAX)
                        .build(), featureLayer)
                .setOutputs(finalLayer)
                .build();

        System.out.println(resNet50Transfer.getNumOutputArrays());
        System.out.println(resNet50Transfer.getOutputLayer(0));
        System.out.println(resNet50Transfer.getOutputLayer(0).conf().getLayer().getLayerName());

        resNet50Transfer.setListeners(
                new ScoreIterationListener(1),
                new PerformanceListener(1),
                new TimeIterationListener((5*classes*100)/batchSize));

        System.out.println("\n --- Training Network ---");

        MultiLayerConfiguration high;
        String modelFilename = null;
        double highest = Double.MIN_VALUE;

        ComputationGraph trained = null;

        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            resNet50Transfer.fit(trainIter);
            System.out.println("epoch: " + i + " score: " + net.score());

            if (net.score() > highest) {
                highest = resNet50Transfer.score();
                System.out.println("epoch: " + i + " highest: " + highest);
                modelFilename = String.format("%s_lowest_err_%f_epoch_%d_RESNet50_transfer.zip", baseDir, highest, i);
                resNet50Transfer.save(new File(modelFilename));
                trained = net.clone();
            }
        }

        trained.setListeners(
                new EvaluativeListener(trainIter, 1),
                new EvaluativeListener(testIter, 1)
        );

        trainIter.reset();
        Evaluation trainEval = trained.evaluate(trainIter);
        Evaluation testEval = trained.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());
    }

    private void setUpTrainTest() throws IOException {

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

    public static void main (String[] args) {
        System.out.println("Hello World");
    }
}
