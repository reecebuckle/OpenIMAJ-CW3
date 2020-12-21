package uk.ac.soton.ecs.cw3;

import gov.sandia.cognition.learning.function.distance.EuclideanDistanceSquaredMetric;
import org.apache.commons.collections.map.HashedMap;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.vfs2.FileSystemException;
import org.geonames.utils.Distance;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.knn.FloatNearestNeighbours;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

import javax.management.AttributeList;
import java.util.*;
import java.util.Map.Entry;

public class TinyImageClassifier {

    private final int SIZE = 16;
    private final int K = 3;

    private VFSGroupDataset<FImage> data;
    private VFSListDataset<FImage> testing;
    private List<String> classes = new ArrayList<>();

    private DoubleNearestNeighboursExact knn;

    public TinyImageClassifier() throws FileSystemException {

        getImages();

        trainTest(80, 20);

    }

    private void trainTest (int trainSize, int testSize) {

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(data, trainSize, 0, testSize);

        GroupedDataset<String, ListDataset<FImage>, FImage> train = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();

        Map<String, double[][]> trainingVectors = extractFeatureVectors(train);
        double [][] vectors = extractFeatureVectors(trainingVectors);
        System.out.println(vectors.length);

        knn = new DoubleNearestNeighboursExact(vectors);

        Map<String, double[][]> testVectors = extractFeatureVectors(test);

        for (String group : testVectors.keySet()) {
            double[][] groupVectors = testVectors.get(group);
            int N = groupVectors.length;

            System.out.println(String.format("%s: #%d", group, testVectors.get(group).length));

            int[][] indices = new int[N][K];
            double[][] distances = new double[N][K];

            knn.searchKNN(testVectors.get(group), K, indices, distances);

            for (int i = 0 ; i < N ; i ++) {
                System.out.print("Indices: ");
                for (int j : indices[i]) System.out.print(j + " ");
                System.out.print("\nClasses: ");
                for (int j : indices[i]) System.out.print(searchClass(vectors[j], trainingVectors) + " ");
                System.out.print("\nDistances: ");
                for (double j : distances[i]) System.out.print(j + " ");
                System.out.println("");
            }


            for (int i = 0 ; i < N ; i ++) {
                Map<String, Integer> classCount = new HashMap<>();
                for (int j : indices[i]) {
                    String neighbour = searchClass(vectors[j], trainingVectors);

                    if (classCount.containsKey(neighbour)) {
                        i = classCount.get(neighbour);
                        classCount.put(neighbour, i+1);
                    } else {
                        classCount.put(neighbour, 1);
                    }
                }
            }

            System.out.println();
        }
    }

    private String searchClass(double[] vector, Map<String, double[][]> mappedVectors) {

        for (Entry<String, double[][]> entry : mappedVectors.entrySet()) {
            for (double[] vec : entry.getValue()) {
                if (Arrays.equals(vec, vector)) return entry.getKey();
            }
        }
        return null;
    }

    private double[][] extractFeatureVectors(Map<String, double[][]> data) {

        int N = 0;
        for (String group : data.keySet()) {
            N += data.get(group).length;
        }

        double[][] vectors = new double[N][SIZE*SIZE];

        int i = 0;
        for (Entry<String, double[][]> entry : data.entrySet()) {
            for (double[] vector : entry.getValue()) {
                vectors[i] = vector;
                i += 1;
            }
        }

        return vectors;
    }

    private void getImages() throws FileSystemException {

        String trainPath = "C:\\Users\\maxmr\\Documents\\Year 3\\Computer Vision\\Handin\\OpenIMAJ-CW3\\OpenIMAJ-CW3\\training";
        String testPath = "C:\\Users\\maxmr\\Documents\\Year 3\\Computer Vision\\Handin\\OpenIMAJ-CW3\\OpenIMAJ-CW3\\testing";

        data = new VFSGroupDataset<>(trainPath, ImageUtilities.FIMAGE_READER);
        testing = new VFSListDataset<>(testPath, ImageUtilities.FIMAGE_READER);

    }

    public Map<String, double[][]> extractFeatureVectors(GroupedDataset<String, ListDataset<FImage>, FImage> data) {

        Map<String, double[][]> groupedVectors = new HashMap<>();

        VectorExtractor vectorExtractor = new VectorExtractor();
        
        for (String group : data.getGroups()) {

            List<double[]> featureVectors = new ArrayList<>();

            for (FImage image : data.get(group)) {
                DoubleFV featureVector = vectorExtractor.extractFeature(image);
                featureVectors.add(featureVector.asDoubleVector());
            }

            groupedVectors.put(group, featureVectors.toArray(new double[][] {}));
        }
        return groupedVectors;
    }

    class VectorExtractor implements FeatureExtractor<DoubleFV, FImage> {

        @Override
        public DoubleFV extractFeature(FImage image) {

            int width = image.getWidth(), height = image.getHeight();

            if (width != height) {
                int size = Math.min(width, height);
                image = image.extractCenter(size, size);
            }

            image.processInplace(new ResizeProcessor(SIZE, SIZE));
            image.processInplace(new MeanCenter());

            DoubleFV feature = new DoubleFV(image.getDoublePixelVector());

            return feature.normaliseFV();
        }
    }


}
