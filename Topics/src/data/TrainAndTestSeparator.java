package data;

import java.io.File;
import java.util.Scanner;

public class TrainAndTestSeparator {
	
	/*
	 * Method loads a query log separating it into a TRAINING and TESTING set
	 * and saves them out to file.
	 * The last (most recent) K queries by each user are used for the TEST data.
	 * If the byPercent parameter is true the last K % of each user's queries are used for the TEST data.
	 * Note that K should be smaller than the minimum number of queries per user
	 */
	
	public static void splitTrainingAndTest(String filename, int K) {
		splitTrainingAndTest(filename, K, false);
	}
			
	public static void splitTrainingAndTest(String filename, int K, boolean byPercent) {
		
		QueryLog train = new QueryLog();
		QueryLog test = new QueryLog();
		int currentK = 0;
		System.out.println("Separating data into N(u)-K" +(byPercent? "%":"")+ " and K"+(byPercent? "%":"")+"...");
		
		try {
			Scanner s = new Scanner(new File(filename+".log"));
			
			int N = s.nextInt();
			int W = s.nextInt();
			int U = s.nextInt();
			int D = s.nextInt();
			
			// TRAINING DATA
			train.N = 0; // we will update the count N as we read in the data
			train.W = W;
			train.U = U;
			train.D = D;
			train.w_uij = new int[U][][];
			train.d_ui  = new int[U][];
			train.t_ui  = new long[U][];
			
			// TESTING DATA
			test.N = 0; 
			test.W = W;
			test.U = U;
			test.D = D;
			test.w_uij = new int[U][][];
			test.d_ui  = new int[U][];
			test.t_ui  = new long[U][]; 
			
			for (int u=0; u<U; u++) {
				
				int I = s.nextInt();
				
				if(byPercent) currentK = (int)((double)I*((double)K/100));
				else currentK = K;
				
				// TRAINING DATA
				train.w_uij[u] = new int[I-currentK][];
				train.d_ui[u]  = new int[I-currentK];
				train.t_ui[u]  = new long[I-currentK];
				
				for (int i=0; i<I-currentK; i++) {
					train.d_ui[u][i] = s.nextInt(); 
					train.t_ui[u][i] = s.nextLong();
					int J = s.nextInt();
					train.N += J; // update word count
					train.w_uij[u][i] = new int[J];
					for (int j=0; j<J; j++) train.w_uij[u][i][j] = s.nextInt();
				}
				
				// TESTING DATA
				// we save the last K queries for each user for testing
				// NOTE: queries are chronologically ordered
				test.w_uij[u] = new int[currentK][];
				test.d_ui[u]  = new int[currentK];
				test.t_ui[u]  = new long[currentK]; 
				
				for (int i=0; i<currentK; i++) {
					test.d_ui[u][i] = s.nextInt(); 
					test.t_ui[u][i] = s.nextLong();
					int J = s.nextInt();
					test.N += J; // update word count
					test.w_uij[u][i] = new int[J];
					for (int j=0; j<J; j++) test.w_uij[u][i][j] = s.nextInt();
				}
				
			}
			
			// check total word count 
			if (N != train.N + test.N) System.err.println("TrainAndTestSeparator: ERROR word count totals don't add up!!");
			
			s.close();
			
			System.out.println("Saving training data ...");
			train.save(filename+"_train_"+K+(byPercent?"p":""));
			
			System.out.println("Saving testing data ...");
			test.save(filename+"_test_"+K+(byPercent?"p":""));
			
		}
		catch (Exception e) {
			e.printStackTrace();
		}		
		
	}

	
	public static void main (String[] args) {
		
		int K;
		boolean byPercent = false;
		
		if(args[0].indexOf('p') == -1)
			K = Integer.parseInt(args[0]);
		else {
			K = Integer.parseInt(args[0].substring(0, args[0].indexOf("p")));
			byPercent = true;
		}
		
		System.out.println("Loading dataset from query log file, K="+K);
		splitTrainingAndTest("aol",K, byPercent);
		
		System.out.println("Finished separating training and test data.");			
		
	}

}
