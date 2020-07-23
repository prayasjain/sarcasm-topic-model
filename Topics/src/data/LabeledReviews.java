package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import utils.QueryTokenizer;


/*
 * 
 * QueryLog.java modified by Aditya
 */
public class LabeledReviews {

	public int[] Z;
	public int N; // total word occurrences
	public int W;  // vocabulary
	public int D;  // number of documents 
	public int S; // number of labels
	public int T_D; // number of test documents

	public int[][] w_ij; // w_ij[i][j] = j'th word in the i'th document
	public ArrayList[] s_i;  // l_i[i] = sentiment label for i'th document
	public int[][] s_ij; // s_ij[i][j] = sentiment of j'th word in the i'th document

	//	public long[][]  t_ui;  // t_ui[u][i] = timestamp for i'th query by the u'th user
	public String[]  l_w;   // l_w[w] = w'th word in the vocabulary
	public static HashMap<String,Integer> hm_l_w;
	public int[]     N_w;   // N_w[w] = count for w'th word in vocab
	public int[]	 N_d;	// N_d[d] = document frequency for w'th word in vocab

	public HashMap<String, Integer> hm_sentiwordlist; // Wordlist containing sentiment words: +1, -1 
	String filename; 

	/* Test corpus */
	public int[][] t_w_ij;	// w_ij[i][j] = j'th word in i'th document of test corpus
	public int[][] t_s_ij;

	public static int sanity_check;
	
	public void loadSentiLexicon(String filename)
	{
		HashMap<String, Integer> hm_senti = new HashMap<String, Integer>();
		try {

			Scanner s = new Scanner(new File(filename));
			while (s.hasNext())
			{
				String str = s.nextLine();
				StringTokenizer st = new StringTokenizer(str," ");
				int num_of_tokens = st.countTokens();

				if (num_of_tokens >= 2)
				{
					String word = st.nextToken();
					String numeric = "";
					
					while (st.hasMoreTokens())
					{
						numeric = numeric+st.nextToken();
					}
					int polarity = Integer.parseInt(numeric);

					hm_senti.put(word,polarity);
				}
			}

			this.hm_sentiwordlist = hm_senti;
		}
		catch (Exception e) {
			e.printStackTrace();
		}	

		System.out.println("Completed loading " + this.hm_sentiwordlist.size() + " words in sentiment word list.");

	}

	public void load(String filename, int num_docs, int num_labels,int words_thresh, boolean retain_sent, boolean hasDocLabels, boolean hasSentimentLabels) {
		load(filename,false, num_docs, num_labels,words_thresh, retain_sent, hasDocLabels, hasSentimentLabels);
		System.out.println("hayeaa");
	}

	public ArrayList<String> getLowFreqWords(String filename,int num_docs,int num_labels,int words_thresh) throws FileNotFoundException
	{
		HashMap<String, Integer> hm_l_w; // Hashmap for vocabulary
		HashMap<String, String> hm_l_w_count; // Hashmap for vocabulary count

		BufferedReader s = new BufferedReader(new FileReader(new File(filename)));

		// We shall count N on the way.. N = s.nextInt();

		//	U = s.nextInt();
		D = num_docs;
		N = 0;
		S = num_labels;

		//l_w = new String[W];
		int vocab_index = -1;
		hm_l_w = new HashMap<String, Integer>();
		hm_l_w_count = new HashMap<String, String>();
		QueryTokenizer qt = new QueryTokenizer(true,  true,  true);


		for (int d=0; d<D; d++) {

			/*	if (!s.())
			{
				System.out.println("ERROR AT d = "+d);
				return null;
			} */
			String I = "";
			try {
				I = s.readLine();
			} catch (IOException e) {
				System.out.println("d = "+d);
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if (I==null || I.trim().equals(""))
				continue;

			StringTokenizer st = new StringTokenizer(I," ");
			String name = "", label = "", review = "";

			if (st.countTokens() < 2)
			{
				
				D--;
				continue;
			}


			// name = st.nextToken().trim();
			//label = st.nextToken().trim();
			review = st.nextToken().trim();
			
			/*
			 * This tokenizes the strings. "False", "True", "True" indicate
			 * whether stemming, stop word removal, etc. needs to be done.
			 */
			review = qt.tokenizeAndReturnString(review, false, true, true);


			st = new StringTokenizer(review," ");


			/*
			 * For each word in a review,
			 */
			while (st.hasMoreTokens())
			{
				String curr_word = st.nextToken();

				/*
				 * If word was enlisted in hm_l_w, increment the count. Else, add it to the 
				 * vocabulary. (vocab_index maintains the maximum current index)
				 */
				if (hm_l_w.containsKey(curr_word))
				{

					//hm_l_w_count.put(curr_word,Integer.parseInt((String)(hm_l_w_count.get(curr_word).trim())+1);
					hm_l_w_count.put(curr_word,Integer.toString(Integer.parseInt((String)hm_l_w_count.get(curr_word))+1));

				}
				else
				{
					vocab_index++;
					hm_l_w.put(curr_word,vocab_index);
					hm_l_w_count.put(curr_word,"1");

				}
			}

		}

		try {
			s.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		ArrayList <String> wordsToRemove = new ArrayList<String>();

		/* Populate l_w and N_w from hm_l_w and hm_l_w_count that have been recorded
		 * during load operation.
		 * 
		 */
		Iterator it = hm_l_w.entrySet().iterator();
		System.out.println("Vocabulary size= "+ hm_l_w.size());

		while (it.hasNext()) {
			Map.Entry pairs = (Map.Entry)it.next();

			if (Integer.parseInt((String) hm_l_w_count.get((String)pairs.getKey())) <= words_thresh)
				wordsToRemove.add((String) pairs.getKey());
			it.remove(); // avoids a ConcurrentModificationException
		}

		/* Hack hack hack! */
		wordsToRemove.add("film");
		wordsToRemove.add("films");
		wordsToRemove.add("movie");
		wordsToRemove.add("movies");

		return wordsToRemove;

	}

	public void load(String filename, boolean loadTimestamps, int num_docs, int num_labels, int words_thresh, boolean retainSentences, boolean hasDocLabels, boolean hasSentenceLabels) {
		this.filename = filename;
		try {

			System.out.println("Loading started...");

			HashMap<String, String> hm_l_w_count; // Hashmap for vocabulary count


			// We shall count N on the way.. N = s.nextInt();

			//	U = s.nextInt();
			D = num_docs;
			N = 0;
			S = num_labels;

			Z = new int[S];

			w_ij = new int[D][];

			if (hasSentenceLabels)
			{
				s_ij = new int[D][];
			}

			if (hasDocLabels)
			{
				s_i  = new ArrayList[S+1];

				for (int i = 0; i < S+1; i++)
					s_i[i] = new ArrayList<Integer>();
			}

			//l_w = new String[W];
			int vocab_index = -1;
			hm_l_w = new HashMap<String, Integer>();
			hm_l_w_count = new HashMap<String, String>();

			ArrayList<String> words_to_remove = getLowFreqWords(filename, num_docs, num_labels, words_thresh);
			QueryTokenizer qt = new QueryTokenizer(true,  true,  true);

			System.out.println("Words occurring less than "+words_thresh+" times will be removed. The list has been compiled.");
			BufferedReader s = new BufferedReader(new FileReader(new File(filename)));


			System.out.print("Loading document #");
			/*
			 * Repeat for all documents.
			 */
			for (int d=0; d<D; d++) {
				if (d%1000 == 0)
					System.out.print(d+"..");
				if (d%10000 == 0)
					System.out.println("");

				String I = s.readLine();
				if (I== null || I.trim().equals(""))
					continue;


				/*
				 * Get the name of file, label and review contents. "name" is currently not
				 * used at all. 
				 */
				StringTokenizer st = new StringTokenizer(I," ");

				if (st.countTokens() < 2)
				{
					
					continue;
				}

				String name = "", label = "", review = "";
			//	name = st.nextToken().trim();
			//	label = st.nextToken().trim();
				review = st.nextToken().trim();



				review = review.toLowerCase();
				/*
				 * This tokenizes the strings. "False", "True", "True" indicate
				 * whether stemming, stop word removal, etc. needs to be done.
				 */
				if (retainSentences)
				{
					review = review.replace("."," <BR>");
					review = review.replaceAll("^ +| +$|( )+", "$1");
					review = review.replaceAll("<BR> <BR>", "<BR>");

				}
				/* Remove all HTML tags */
				review = review.replaceAll("<.*>", "");

				/* And all &nbsp; style tags */
				review = review.replaceAll("&.*;","");

				
				/* While you are at it. Also, record the sentence labels if you must.*/
				ArrayList<Integer> sentimentSequence = new ArrayList<Integer>();


				if (hasSentenceLabels)
				{
					st = new StringTokenizer(review, " ");

					String s_prev = "", s_cur = "";
					while (st.hasMoreTokens())
					{

						s_cur = st.nextToken();

						if (hasSentenceLabels && s_cur.equals("<BR>"))
						{
							int curr_val = Integer.parseInt(s_prev.replace("#",""));

							if (curr_val == 0)
								sentimentSequence.add(0);
							else if (curr_val > 0)
								sentimentSequence.add(1);
							else
								sentimentSequence.add(2);
						}


						s_prev = s_cur;
					}


				}
				/*
				 * Hack: POS Tagged words are of the form: word_tag. This line concatenates the two. 
				 * This ensures that the tokenize function gives us "word tag".
				 */
				review = review.replaceAll("_","");
				
				review = qt.tokenizeAndReturnString(review, false, true, true);
				
				/*
				 * Remove all low-frequency words
				 */

				for(String rem_word:words_to_remove)
				{
					review = review.replaceAll(" "+ rem_word +" "," ");
				}

				/*
				 * Based on document label assigned, add document number to the corresponding
				 * ArrayList. s_i corresponds to three arraylists, one for each label.
				 */
				if (hasDocLabels)
				{
					if (label.equals("P"))

						s_i[1].add(d);
					else if (label.equals("N"))
						s_i[2].add(d);
					else
						s_i[0].add(d);

				}
				/*
				 * Calculate length of a document. This is to define length of array
				 * w_ij[d].
				 * 

				 */
				st = new StringTokenizer(review, " ");
				int doc_length = 0;
				int num_sentences = 0; // Not used currently. Just a local count.


				
				/* Count document length */
				while (st.hasMoreTokens())
				{

					st.nextToken();
					doc_length++;

				}

				if (hasSentenceLabels)
				{
					s_ij[d] = new int[doc_length];
				}

				w_ij[d] = new int[doc_length];

				st = new StringTokenizer(review," ");
				int curr_j = 0;

				int curr_sentence = 0; // This is used only if hasSentenceLabels is on.
				/*
				 * For each word in a review,
				 */

				while (st.hasMoreTokens())
				{
					String curr_word = st.nextToken();
					N++;
					int pos = -1;

					/*
					 * If word was enlisted in hm_l_w, increment the count. Else, add it to the 
					 * vocabulary. (vocab_index maintains the maximum current index)
					 */
					if (hm_l_w.containsKey(curr_word))
					{

						//hm_l_w_count.put(curr_word,Integer.parseInt((String)(hm_l_w_count.get(curr_word).trim())+1);
						hm_l_w_count.put(curr_word,Integer.toString(Integer.parseInt((String)hm_l_w_count.get(curr_word))+1));
						pos = (Integer) hm_l_w.get(curr_word);
					}
					else
					{
						vocab_index++;
						hm_l_w.put(curr_word,vocab_index);
						hm_l_w_count.put(curr_word,"1");
						pos = (Integer) hm_l_w.get(curr_word);
					}

					/*
					 * This is essentially the operation you wanted to do!
					 * For each sentence, get the corresponding label.
					 */
					if (hasSentenceLabels)
					{


						s_ij[d][curr_j] = sentimentSequence.get(curr_sentence);

						if (curr_word.equals("BR") || curr_word.equals("br"))
						{
							curr_sentence++;
						}
					}

					w_ij[d][curr_j] = pos;
					curr_j++;

				}

			}


			s.close();
			W = vocab_index+1;
			l_w = new String[vocab_index+1];
			N_w = new int[vocab_index+1];

			/* Populate l_w and N_w from hm_l_w and hm_l_w_count that have been recorded
			 * during load operation.
			 * 
			 */
			Iterator it = hm_l_w.entrySet().iterator();
			System.out.println("Tally="+hm_l_w.size()+" "+ hm_l_w_count.size());
			sanity_check = hm_l_w.size();
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry)it.next();

				l_w[(Integer)pairs.getValue()] = (String) pairs.getKey();
				N_w[(Integer)pairs.getValue()] =  Integer.parseInt((String) hm_l_w_count.get((String)pairs.getKey()));
				it.remove(); // avoids a ConcurrentModificationException
			}

			System.out.println("Loading complete.");
		}
		catch (Exception e) {
			e.printStackTrace();
		}			
	}

	public void loadTestCorpus(String filename, boolean loadTimestamps, int num_docs, int num_labels, int words_thresh, boolean retainSentences, boolean hasDocLabels, boolean hasSentenceLabels) {
		this.filename = filename;
		try {

			System.out.println("Test Loading started...");

			// We shall count N on the way.. N = s.nextInt();

			//	U = s.nextInt();
			T_D = num_docs;
			N = 0;
			S = num_labels;

			Z = new int[S];

			t_w_ij = new int[T_D][];

			if (hasSentenceLabels)
			{
				s_ij = new int[T_D][];
			}

			if (hasDocLabels)
			{
				s_i  = new ArrayList[S+1];

				for (int i = 0; i < S+1; i++)
					s_i[i] = new ArrayList<Integer>();
			}

			//l_w = new String[W];
			int vocab_index = -1;


			QueryTokenizer qt = new QueryTokenizer(true,  true,  true);

			BufferedReader s = new BufferedReader(new FileReader(new File(filename)));


			/*
			 * Load vocabulary again
			 */
			hm_l_w = new HashMap<String,Integer>();
			
			
			for (int i =0 ; i < l_w.length; i++)
			{
				hm_l_w.put(l_w[i],i);
			}
			
			if(l_w.length == sanity_check)
				System.out.println("Sanity check. You loaded and retrieved vocabulary of "+sanity_check);
			else
				System.out.println("Sanity check. You retrieved vocabulary of "+l_w.length+" while you loaded "+sanity_check);
			
			System.out.print("Loading test document #");
			/*
			 * Repeat for all documents.
			 */
			
			
			for (int d=0; d<T_D; d++) {

				if (d%1000 == 0)
					System.out.print(d+"..");
				if (d%10000 == 0)
					System.out.println("");

				String I = s.readLine();
				if (I==null || I.trim().equals(""))
				{
					continue;
				}


				/*
				 * Get the name of file, label and review contents. "name" is currently not
				 * used at all. 
				 */
				StringTokenizer st = new StringTokenizer(I," ");

				if (st.countTokens() < 2)
				{
					
					T_D--;
					continue;
				}

				String name = "", label = "", review = "";
			//	name = st.nextToken().trim();
			//	label = st.nextToken().trim();
				review = st.nextToken().trim();



				review = review.toLowerCase();
				/*
				 * This tokenizes the strings. "False", "True", "True" indicate
				 * whether stemming, stop word removal, etc. needs to be done.
				 */
				if (retainSentences)
				{
					review = review.replace("."," <BR>");
					review = review.replaceAll("^ +| +$|( )+", "$1");
					review = review.replaceAll("<BR> <BR>", "<BR>");

				}
				/* Remove all HTML tags */
				review = review.replaceAll("<.*>", "");

				/* And all &nbsp; style tags */
				review = review.replaceAll("&.*;","");

				/* While you are at it. Also, record the sentence labels if you must.*/
				ArrayList<Integer> sentimentSequence = new ArrayList<Integer>();


				if (hasSentenceLabels)
				{
					st = new StringTokenizer(review, " ");

					String s_prev = "", s_cur = "";
					while (st.hasMoreTokens())
					{

						s_cur = st.nextToken();

						if (hasSentenceLabels && s_cur.equals("<BR>"))
						{
							int curr_val = Integer.parseInt(s_prev.replace("#",""));

							if (curr_val == 0)
								sentimentSequence.add(0);
							else if (curr_val > 0)
								sentimentSequence.add(1);
							else
								sentimentSequence.add(2);
						}


						s_prev = s_cur;
					}


				}
				/*
				 * Hack: POS Tagged words are of the form: word_tag. This line concatenates the two. 
				 * This ensures that the tokenize function gives us "word tag".
				 */
				review = review.replaceAll("_","");

				review = qt.tokenizeAndReturnString(review, false, true, true);

				/*
				 * Remove all low-frequency words
				 */


				/*
				 * Based on document label assigned, add document number to the corresponding
				 * ArrayList. s_i corresponds to three arraylists, one for each label.
				 */
				if (hasDocLabels)
				{
					if (label.equals("P"))

						s_i[1].add(d);
					else if (label.equals("N"))
						s_i[2].add(d);
					else
						s_i[0].add(d);

				}
				/*
				 * Calculate length of a document. This is to define length of array
				 * w_ij[d].
				 * 

				 */
				st = new StringTokenizer(review, " ");
				int doc_length = 0;
				int num_sentences = 0; // Not used currently. Just a local count.


				
				/* Count document length */
				while (st.hasMoreTokens())
				{
					st.nextToken();
					doc_length++;
				}

				if (hasSentenceLabels)
				{
					t_s_ij[d] = new int[doc_length];
				}

				t_w_ij[d] = new int[doc_length];

				st = new StringTokenizer(review," ");
				int curr_j = 0;

				int curr_sentence = 0; // This is used only if hasSentenceLabels is on.
				/*
				 * For each word in a review,
				 */

				while (st.hasMoreTokens())
				{
					String curr_word = st.nextToken();
					
					int pos = -1;

					/*
					 * If word was enlisted in hm_l_w, increment the count. Else, add it to the 
					 * vocabulary. (vocab_index maintains the maximum current index)
					 */
					if (hm_l_w.containsKey(curr_word))
					{
						pos = (Integer) hm_l_w.get(curr_word);
					}
					else
					{
						pos = -1;
					}

					t_w_ij[d][curr_j] = pos;
					curr_j++;
					
					/*
					 * This is essentially the operation you wanted to do!
					 * For each sentence, get the corresponding label.
					 */
					if (hasSentenceLabels)
					{


						t_s_ij[d][curr_j] = sentimentSequence.get(curr_sentence);

						if (curr_word.equals("BR") || curr_word.equals("br"))
						{
							curr_sentence++;
						}
					}

					

				}

			}


			s.close();


			System.out.println("Loading complete.");
		}
		catch (Exception e) {
			e.printStackTrace();
		}				
	}


	public void printStats() {

		System.out.println("Labeled corpus statistics:");

		System.out.println("\t # of docs:  "+D);
		System.out.println("\t Vocab size: "+W);
		System.out.println("\t Word occurrences: "+ N);
		System.out.println("\t Number of labels : "+ S);
	}


}
