package utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Scanner;

public class TopicModelUtils {

	public static void printTopTerms(double[][] p_w_z, String[] lexicon) {
		int Z = p_w_z[0].length;
		// cycle over topics
		for (int z=0; z<Z; z++) {
			System.out.println("\nTopic "+z);
			// rank the terms
			double[] p_w = getColumn(p_w_z,z);
			int[] rankedListOfTerms = utils.Sorter.rankedList(p_w);
			// print out the top 5 terms
			int numWordsToPrint = 10;
			for (int i=0; i<numWordsToPrint; i++) {
				int w = rankedListOfTerms[i];
				System.out.printf("%7.3f %s", p_w[w]*100, lexicon[w]);
				System.out.println();
			}
		}
	}
	
	public static void printTopTerms(double[][][] p_s_w_z, String[] lexicon) {
		int S = p_s_w_z.length;
		int Z = p_s_w_z[0][0].length;
		System.out.printf("Sentiment-->              ");
		for (int s=0; s<S; s++) System.out.printf("%-4d                 ",s);
		System.out.println();
		// cycle over topics
		for (int z=0; z<Z; z++) {
			// rank the terms
			double[][] p_s_w = new double[S][];
			int[][] rankedListOfTerms = new int[S][];
			for (int s=0; s<S; s++) {
				p_s_w[s] = getColumn(p_s_w_z[s],z);
			    rankedListOfTerms[s] = utils.Sorter.rankedList(p_s_w[s]);
			}    
			// print out the top 5 terms
			int numWordsToPrint = 5;
			for (int i=0; i<numWordsToPrint; i++) {
				if(i==0) {
					System.out.println();
					System.out.printf("Topic %-3d              ",z);
				}
				else
					System.out.printf("                       ");
				
				for (int s=0; s<S; s++) {
					int w = rankedListOfTerms[s][i];
					System.out.printf("%7.3f %-10.10s          ", p_s_w[s][w]*100, lexicon[w]);
		                    		  
					
				}	
				System.out.println();
			}
		}
	}
	
	public static void printCountSentiWords(double[][] p_w_z, String[] lexicon,HashMap hm_senti) {
		int Z = p_w_z[0].length;
		int pos_count = 0, neg_count = 0, total_count = 0;
		if (hm_senti == null)
		{
			System.out.println("Warning: No sentiment lexicon attached. Cannot count sentiment words.");
			return;
		}
		System.out.println("Number of words by sentiment for each topic are as follows.");
		System.out.println("Format: (pos count, neg count, total count)");
		// cycle over topics
		for (int z=0; z<Z; z++) {
			
			pos_count = 0;
			neg_count = 0;
			total_count = 0;
			
			// rank the terms
			double[] p_w = getColumn(p_w_z,z);
			int[] rankedListOfTerms = utils.Sorter.rankedList(p_w);
			// print out the top 20 terms
			for (int i=0; i< 10; i++) {
				total_count++;
				int w = rankedListOfTerms[i];
				
				
				
				if (hm_senti.containsKey(lexicon[w]))
				{	
					if((Integer) hm_senti.get(lexicon[w]) == 1)
						pos_count++;
				
					else if ((Integer) hm_senti.get(lexicon[w]) == 0)
						neg_count++;
				}
			}
			System.out.println("Topic "+z+": " + pos_count + " "+ neg_count + " "+ total_count);
		}
	}
	
	public static double[] getColumn(double[][] matrix, int columnIndex) {
		int M = matrix.length;
		double[] column = new double[M];
		for (int i=0; i<M; i++) 
			column[i] = matrix[i][columnIndex];
		return column;
	}
	
	
	public static void saveMatrix(double[][] matrix, String filename) {
		try {
			PrintWriter p = new PrintWriter(new FileOutputStream(filename));
			p.println(matrix.length); // save row and column count
			p.println(matrix[0].length);
			for (double[] row: matrix) { // save data
				boolean first = true;
				for (double val: row) {
					if (first) first = false;
					else p.print(' '); 
					p.print(val);
				}
				p.println();
			}
			p.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}		

	}
	
	
	public static double[][] loadMatrix(String filename) {
		double[][] matrix = null;
		try {
			Scanner s = new Scanner(new File(filename));
			int M = s.nextInt(); // number of rows
			int N = s.nextInt(); // number of columns
			matrix = new double[M][N];
			for (int i=0; i<M; i++)
				for (int j=0; j<N; j++)
					matrix[i][j] = s.nextDouble(); 
			s.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return matrix;
	}

	
	public static void saveNonRectMatrix(double[][] matrix, String filename) {
		try {
			PrintWriter p = new PrintWriter(new FileOutputStream(filename));
			p.println(matrix.length); // row count
			p.println(-1);            // -1 indicates variable column count 
			for(double[] row: matrix) {
				p.println(row.length);
				boolean first = true;
				for (double val: row) {
					if (first) first = false;
					else p.print(' '); 
					p.print(val);
				}
				p.println();
			}
			p.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}	
	}

	
	public static double[][] loadNonRectMatrix(String filename) {
		double[][] matrix = null;
		try {
			Scanner s = new Scanner(new File(filename));
			int currentN = 0;
			int M = s.nextInt(); // number of rows
			int check = s.nextInt(); // check that this is a non-rect matrix
			if(check == -1) {
			matrix = new double[M][];
			for (int i=0; i<M; i++) {
				currentN = s.nextInt();
				matrix[i] = new double[currentN];
				
				for (int j=0; j<currentN; j++)
					matrix[i][j] = s.nextDouble(); 
			}
			s.close();
			}
			else
				return null;
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return matrix;
	}
	
	
	public static void saveVector(double[] vector, String filename) {
		try {
			PrintWriter p = new PrintWriter(new FileOutputStream(filename));
			p.println(vector.length); // column count
			boolean first = true;
			for (double val: vector) {
				if (first) first = false;
				else p.print(' '); 
				p.print(val);
			}
			p.println();
			p.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}		

	}
	
	
	public static double[] loadVector(String filename) {
		double[] vector = null;
		try {
			Scanner s = new Scanner(new File(filename));
			int N = s.nextInt(); // number of columns
			vector = new double[N];
			for (int j=0; j<N; j++)
				vector[j] = s.nextDouble(); 
			s.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return vector;
	}


	public static void save3dMatrix(double[][][] matrix, String filename) {
		try {
			PrintWriter p = new PrintWriter(new FileOutputStream(filename));
			p.println(matrix.length);       // rows
			p.println(matrix[0].length);    // columns
			p.println(matrix[0][0].length); // depth
			for (double[][] row: matrix) { // save data
				for (double[] column: row) {
					boolean first = true;
					for (double val: column) {
						if (first) first = false;
						else p.print(' '); 
						p.print(val);
					}
					p.println();
				}	
			}
			p.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}		

	}
	
	
	public static double[][][] load3dMatrix(String filename) {
		double[][][] matrix = null;
		try {
			Scanner s = new Scanner(new File(filename));
			int M = s.nextInt(); // number of rows
			int N = s.nextInt(); // number of columns
			int O = s.nextInt(); // depth
			matrix = new double[M][N][O];
			for (int i=0; i<M; i++)
				for (int j=0; j<N; j++)
					for (int k=0; k<O; k++)
						matrix[i][j][k] = s.nextDouble(); 
			s.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return matrix;
	}

	
	public static Sparse3dMatrix loadSparse3dMatrix(String filename, double threshold) {
		
		Sparse3dMatrix matrix = null;
		try {
			Scanner s = new Scanner(new File(filename));
			int M = s.nextInt(); // number of rows
			int N = s.nextInt(); // number of columns
			int O = s.nextInt(); // depth
		
			double min = threshold/M;
			
			//System.out.println("TopicModelUtils: loading sparse 3d matrix: "+filename+" with minimum value: "+min);
			//System.out.println("Size: M="+M+", N="+N+", O="+O);
			
			matrix = new Sparse3dMatrix(M*N);
			for (int i=0; i<M; i++)
				for (int j=0; j<N; j++)
					for (int k=0; k<O; k++) {
						double val = s.nextDouble();
						if (val > min) matrix.add(i, j, k, val);
					}
			s.close();
			
			//System.out.println("TopicModelUtils: finished loading sparse 3d matrix");
			//System.out.println("Size: "+M+"*"+N+"*"+O+"="+M*N*O+", sparsity factor: "+matrix.x.size()/((double) M*N*O));
				
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		return matrix;
	}

	

}
