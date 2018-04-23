package ca.ooda.fushia.NLP.multigpu;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.deeplearning4j.util.ModelSerializer;
import java.io.File;

public class W2VecSentimentRNNgpu {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(W2VecSentimentRNNgpu.class);

    public static void main(String[] args) throws Exception {

        //Load the model
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork("/home/ouledsmo/Desktop/MyMultiLayerNetwork.zip");

        /** Location to save and extract the training/testing data */
        final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
        int truncateReviewsToLength = 10;  //Truncate reviews with length (# words) greater than this
        int batchSize = 64;     //Number of examples in each minibatch
        final String WORD_VECTORS_PATH = "/home/ouledsmo/Desktop/code/dl4j-examples/dl4j-examples/src/main/resources/GoogleNews-vectors-negative300.bin.gz";

        //After training: load a single example and generate predictions
        String firstPositiveReview = " Please can anyone help me, I'm kind of stuck in an elevator"; // FileUtils.readFileToString(firstPositiveReviewFile);

        //DataSetIterators for training and testing respectively
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

        INDArray features = test.loadFeaturesFromString(firstPositiveReview, truncateReviewsToLength);
        INDArray networkOutput = net.output(features);
        int timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("First positive review: \n" + firstPositiveReview);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Example complete -----");

    }

}
