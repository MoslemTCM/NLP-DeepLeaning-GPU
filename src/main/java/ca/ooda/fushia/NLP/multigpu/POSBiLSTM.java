package ca.ooda.fushia.NLP.multigpu;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class POSBiLSTM {

    // Gets Path to train Text file
    File trainfile = new File (new ClassPathResource("/POS/train.txt").getFile().getAbsolutePath());
    // Gets Path to train Text file
    File testfile = new File (new ClassPathResource("/POS/test.txt").getFile().getAbsolutePath());

    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
    int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    /** Location to save and extract the training/testing data */
    public static final String DATA_PATH = ("/home/ouledsmo/Desktop/data/dl4j_w2vSentiment/");
    /** Location (local file system) for the Google News vectors. Set this manually. */
    public static final String WORD_VECTORS_PATH = "/home/ouledsmo/Desktop/code/dl4j-examples/dl4j-examples/src/main/resources/GoogleNews-vectors-negative300.bin.gz";

    public POSBiLSTM() throws FileNotFoundException {
    }

    public void ReadData() throws IOException {

        //First: load sentence to String.
        int totalexample = 0;
        int maxLength = 0;
        Vector<Vector<String>> vector = new Vector<>();
        Vector<Vector<String>> tag = new Vector<>();
        Vector temp = new Vector<>();
        Vector temp1 = new Vector<>();
        BufferedReader br = new BufferedReader(new FileReader(trainfile));
        String line;
        while ((line = br.readLine()) != null) {
            if (!line.isEmpty()){
            String[] mot = line.split("\t");
            System.out.println(mot[0] +  "       " + mot[1]);
            temp.add(mot[0]);
            temp1.add(mot[1]);
            }
            else if (line.isEmpty()){
            totalexample++;
            maxLength = Math.max(maxLength,temp.size());
            vector.add(temp);
            tag.add(temp1);
            temp.clear();
            temp1.clear();
            }
        }
        br.close();

        System.out.println(vector.size());
        System.out.println(tag.size());
        System.out.println(maxLength);
        System.out.println(totalexample);

        //Create data for training
        //Here: we have vector.size() examples of varying lengths
        INDArray features = Nd4j.create(new int[]{vector.size(), vectorSize, maxLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{vector.size(), 8, maxLength}, 'f');    //Two labels: positive or negative

        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(vector.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(vector.size(), maxLength);
    }

    public static void main(String[] args) throws IOException, InterruptedException {

    int batchSize = 64;     //Number of examples in each minibatch
    int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
    int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
    int truncateReviewsToLength = 15;  //Truncate reviews with length (# words) greater than this
    final int seed = 0;     //Seed for reproducibility

        POSBiLSTM p = new POSBiLSTM();
        p.ReadData();

    /*
    Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

    //Set up network configuration
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .updater(new Adam(2e-2))
            .l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
            .list()
            .layer(0, new Bidirectional(Bidirectional.Mode.ADD, new GravesLSTM.Builder()
                    .nIn(vectorSize).nOut(256).activation(Activation.TANH).build()))
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .pretrain(false).backprop(true).build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

    //DataSetIterators for training and testing respectively
    SentimentExampleIterator train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
    SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

        System.out.println("Starting training");
        for (int i = 0; i < nEpochs; i++) {
        net.fit(train);
        train.reset();
        System.out.println("Epoch " + i + " complete. Starting evaluation:");

        //Run evaluation. This is on 25k reviews, so can take some time
        Evaluation evaluation = net.evaluate(test);
        System.out.println(evaluation.stats());
    }

    //After training: load a single example and generate predictions
    File firstPositiveReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/pos/Output12587.txt"));
    String firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile);

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

     */

    }

}
