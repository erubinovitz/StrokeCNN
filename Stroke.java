package org.deeplearning4j.stroke;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Stroke {
	
    private static Logger log = LoggerFactory.getLogger(Stroke.class);

    public static void main(String[] args) throws Exception {
   
        final int numRows =1;
        final int numColumns = 225; //numFeatures
        //155;
        int outputNum = 7; // number of output classes (0-6 right now)
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 400; // number of epochs to perform 
        int nChannels=1; //stays at 1
        int resultsMod=1; //stays at 1
        int numSamplesPredict=9202; 
        //45925; // # prediction samples
        int numSamplesTest=3944;
        // 21228; // # test samples
        int batchSize=150;  // batch size for each epoch. Too large = linear, too small = zig-zaggy
        String mainPath="src/main/java/org/deeplearning4j/stroke";
        String trainPath = mainPath+"/3m_nomissingTrain.csv";
        //		"/TSR_curated_2018-07-20Train.csv";
        String testPath = mainPath+"/3m_nomissingTest.csv";
        //		"/TSR_curated_2018-07-20Test.csv";
        String resultsPath =mainPath+"/testResults"; // where to write the results of test predictions
    

        Evaluation eval = new Evaluation(outputNum);           
        UIServer uiServer = UIServer.getInstance();        
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later        
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
      try {
        long exitDelay=10000; //time in ms for the localhost to stay up before closing; if not closed,
        					  //program cannot be run again without manually shutting down server from cmd
        Scanner sc = new Scanner (new File(trainPath));
        String[] scanned = new String[numSamplesPredict];
        int count=0;
        for (int i=0; i<scanned.length; i++) {
        	String s = sc.nextLine();
        	if (!s.substring(s.lastIndexOf(",")+1).equals(("NA"))) {
        		scanned[count]=s;
        		count++;
        	}
        	
        }

        log.info("Build model....");

        int LC=0; //layercounter, dont touch
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .l2(0.00002)  
                .weightInit(WeightInit.XAVIER)               
                .updater(new Nesterovs(Math.pow(Math.E, -3),.9))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()              
                .layer(LC++, new ConvolutionLayer.Builder(1, 3) // kernel size (right should be 3, 5, or 7, left must be 1 )
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels) //nChannels
                        .stride(1, 1)
                        .nOut(8) //numOutputFilters 
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(16)
                		.activation(Activation.RELU)
                		.build()
                		)
        /*        .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(FC+=FI)
                		.activation(Activation.RELU)
                		.build()
                		)*/
                .layer(LC++, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(1,1) //Left must be 1
                        .stride(2,2)
                        .build())
                .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(24)
                		.activation(Activation.RELU)
                		.build()
                		)
                .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(32)
                		.activation(Activation.RELU)
                		.build()
                		)
        /*        .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(FC+=FI)
                		.activation(Activation.RELU)
                		.build()
                		)*/
                .layer(LC++, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(1,1) //Left must be 1
                        .stride(2,2)
                        .build())
/*                .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(FC+=FI)
                		.activation(Activation.RELU)
                		.build()
                		)
                .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(FC+=FI)
                		.activation(Activation.RELU)
                		.build()
                		)
                .layer(LC++, new ConvolutionLayer.Builder(1,3)
                		.stride(1,1)
                		.nOut(FC+=FI)
                		.activation(Activation.RELU)
                		.build()
                		)
                .layer(LC++, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(1,1) //Left must be 1
                        .stride(2,2)
                        .build())*/             
             /*   .layer(LC++, new ConvolutionLayer.Builder(1, 1)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(3)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(LC++, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(1,1)
                        .stride(2,2)
                        .build())*/
/*                .layer(LC++, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(600) 
                        	//308 57.53
                        .build())*/
                .layer(LC++, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(250) //numOutputFilters 
                        .build())
                .layer(LC++, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(164) //numOutputFilters 
                        .build()) 
                .layer(LC++, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum) //num output classes, 6 for 0-5
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(numRows,numColumns,nChannels)) //See note below
                .backprop(true).pretrain(false).build();
        
        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)

        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */
   
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();    
        model.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(100));     
        log.info("Train model....");
        /*Handles batches and datafitting*/
        int x=0;      
        for( int i=0; i<=numEpochs; i++ ){
        	double[] vals = new double [numColumns];
        	 int c1=0;
            NDArray nums = new NDArray(batchSize,numColumns);
            NDArray numsLabels = new NDArray(batchSize,outputNum);
            while (true) {         
            	if (x==count)
            		x=0;
            	String s1 = scanned[x++];
            	String[] s = s1.split(",");
            	for (int j=0; j<s.length; j++) {
            		
            		if (j!=s.length-1) {
            			if (s[j].equals("NA")) {
            				nums.putScalar(c1,j,vals[j]/(c1+1));
            				vals[j]+=vals[j]/(c1+1);
            			}
            			else {
            				nums.putScalar(c1,j,Double.parseDouble(s[j]));
            				vals[j]+=Double.parseDouble(s[j]);
            			}
            		}
            		else {
            			if (s[j].equals("NA")) {c1--; break;}
            			numsLabels.putScalar(c1, (int)Math.rint(resultsMod*Double.parseDouble(s[j])),1);
            			break;
            		}
            	}
            	c1++;
            	if (c1%batchSize==0||c1>=numSamplesPredict) break;            	
            }                      
            DataSet data =  new DataSet(nums,numsLabels);            
        	model.fit(data);       	
        }
        sc = new Scanner (new File(testPath));
        NDArray numsTest = new NDArray(numSamplesTest,numColumns);
        NDArray numsLabelsTest = new NDArray(numSamplesTest,outputNum);
        int c=0;
        double[] vals = new double [numColumns];
        while (sc.hasNextLine()) {
        	String[] s = sc.nextLine().split(",");
        	for (int j=0; j<=numColumns; j++) {
        		if (j!=numColumns) {        
        			if (s[j].equals(("NA"))) {
        					numsTest.putScalar(c, j,vals[j]/(c+1));
        					vals[j]+=vals[j]/(c+1);
        			}
        			else {
        				numsTest.putScalar(c,j,Double.parseDouble(s[j]));
        				vals[j]+=Double.parseDouble(s[j]);
        			}
        		}
        		else {
        			if (s[j].equals("NA")) break;
        			
        			numsLabelsTest.putScalar(c, (int)Math.rint(resultsMod*Double.parseDouble(s[j])) ,1);
        		}
        	}
        	c++;
        }
        DataSet dataTest =  new DataSet(numsTest,numsLabelsTest);    	
        log.info("Evaluate model....");
        eval.reset();
        INDArray output = model.output(dataTest.getFeatureMatrix()); //get the networks prediction  
        eval.eval(dataTest.getLabels(), output); 
        log.info(eval.stats());
        double[][] a=output.toDoubleMatrix();
        try {
	        PrintWriter pw = new PrintWriter(new File(resultsPath));
	        System.out.println("Predictions:");
	        double[] maxes = new double[6];
	        for (int i=0; i<a.length; i++) {
	        	double max=a[i][0];
	        	int maxIndex=0;
	        	for (int j=1; j<6; j++)
	        		if (a[i][j]>max) {
	        			max=a[i][j];
	        			maxIndex=j;
	        		}
	        	maxes[maxIndex]++;
	        //	System.out.println(maxIndex);
	        	pw.write(Integer.toString(maxIndex)+"\n");
	        }
	        
	        pw.flush();
	        pw.close();
	   }
       catch (Exception e) {
    	   
       }
        long time = System.currentTimeMillis();
        while (System.currentTimeMillis()-time<=exitDelay)
        	continue;
        uiServer.stop();
        
        log.info("****************Example finished********************");
      }
        catch (Exception e ) {
      	  uiServer.stop();
      	  e.printStackTrace();
        }
          
}
}