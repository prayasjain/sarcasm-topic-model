package data;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Scanner;
import java.util.ArrayList;

public class QueryLog {

	public long N; // total word occurrences
	public int W;  // vocabulary
	public int U;  // number of users
	public int D;  // number of documents (urls)
	
	public int[][][] w_uij; // w_uij[u][i][j] = j'th word in the i'th query by the u'th user
	public int[][]   d_ui;  // d_ui[u][i] = document for i'th query by the u'th user
	
	public long[][]  t_ui;  // t_ui[u][i] = timestamp for i'th query by the u'th user
	public String[]  l_w;   // l_w[w] = w'th word in the vocabulary
	public int[]     N_w;   // N_w[w] = count for w'th word in vocab
	
	String filename; 
	
	public void save(String filename) {
		try {
			PrintWriter p = new PrintWriter(new FileOutputStream(filename+".log"));
			p.println(N);
			p.println(W);
			p.println(U);
			p.println(D);
			for (int u=0; u<U; u++) {
				int I = d_ui[u].length;
				p.println(I);
				for (int i=0; i<I; i++) {
					int J = w_uij[u][i].length;
					p.print(d_ui[u][i] +" "+ t_ui[u][i] +" "+ J); 
					for (int j=0; j<J; j++) p.print(" "+w_uij[u][i][j]);
					p.println();
				}
			}
			p.close();
			
			if (l_w != null) {
				// save also the lexicon
				p = new PrintWriter(new FileOutputStream(filename+".lex"));
				p.println(W);
				for (int w=0; w<W; w++) p.println(l_w[w] +" "+ N_w[w]);
				p.close();
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	
	public void load(String filename) {
		load(filename,false);
	}
	
	
	public void load(String filename, boolean loadTimestamps) {
		this.filename = filename;
		try {
			Scanner s = new Scanner(new File(filename+".log"));
			N = s.nextInt();
			W = s.nextInt();
			U = s.nextInt();
			D = s.nextInt();
			w_uij = new int[U][][];
			d_ui  = new int[U][];
			if (loadTimestamps) t_ui  = new long[U][];
			for (int u=0; u<U; u++) {
				int I = s.nextInt();
				w_uij[u] = new int[I][];
				d_ui[u]  = new int[I];
				if (loadTimestamps) t_ui[u]  = new long[I];
				for (int i=0; i<I; i++) {
					d_ui[u][i] = s.nextInt(); 
					if (loadTimestamps) t_ui[u][i] = s.nextLong();
					else s.nextLong(); // skip timestamp
					int J = s.nextInt();
					w_uij[u][i] = new int[J];
					for (int j=0; j<J; j++) w_uij[u][i][j] = s.nextInt();
				}
			}
			s.close();
		}
		catch (Exception e) {
			e.printStackTrace();
		}				
	}
	
	
	public String[] loadLexicon() {
		if (l_w == null) {
			// remove "_train_X" or "_test_X" from end of filename:
			if (filename.indexOf("_t") != -1) filename = filename.substring(0, filename.indexOf("_t"));
			try {
				Scanner s = new Scanner(new File(filename+".lex"));
				W = s.nextInt();
				l_w  = new String[W];
				N_w  = new int[W];
				for (int w=0; w<W; w++) {
					l_w[w] = s.next();
					N_w[w] = s.nextInt();
				}
				s.close();
			}
			catch (Exception e) {
				e.printStackTrace();
			}
		}
		return l_w;
	}
		
	
	public int[][] projectOverUsers() {
		ArrayList<ArrayList<Integer>> words_di = new ArrayList<ArrayList<Integer>>(D); 
		for (int d=0; d<D; d++) 
			words_di.add(new ArrayList<Integer>());
		for (int u=0; u<U; u++) {
			for (int i=0; i<w_uij[u].length; i++) {
				ArrayList<Integer> words = words_di.get(d_ui[u][i]); 
				for (int w: w_uij[u][i]) 
					words.add(w);
			}
		}	
		int[][] w_di = new int[D][];
		for (int d=0; d<D; d++) {
			ArrayList<Integer> words = words_di.get(d); 
			w_di[d] = new int[words.size()];
			int i=0;
			for (int w: words)
				w_di[d][i++] = w; 
		}	
		return w_di;
	}
	
	
	public void printStats() {
		
		System.out.println("Query log statistics:");
		System.out.println("\t # of users: "+U);
		System.out.println("\t # of docs:  "+D);
		System.out.println("\t Vocab size: "+W);
		System.out.println("\t Word occurrences: "+N);
		
	}
	
	
}
