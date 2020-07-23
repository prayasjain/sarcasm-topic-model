package data;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

import utils.QueryTokenizer;


public class ClickEntropy {
	
	public static void main (String[] args) {
		
		/* STEPS:
		 * 
		 * 1) run sql query to dump contents of clicked_only table to a file:
		 * 	 > cd workspace/tripartite/experiments_queries/clickentropy
		 *   > mysql -umark -ptaipei101 mark < dump_clicked.sql > clicked_queries.txt
		 *   
		 * 2) run stemAndSort procedure on file to parse query terms and order them 
		 *   > cd ../code
		 *   > java -Xmx1g -classpath . data/ClickEntropy stemAndSort ../clickentropy/clicked_queries.txt > ../clickentropy/parsed_queries.txt &
		 *   
		 * 3) sort the lines in the parsed_queries file (note the additional environment variable):
		 *   > cd ../clickentropy
		 *   > export LC_COLLATE=C
		 *   > sort parsed_queries.txt > sorted_queries.txt
		 *   
		 * 4) run calculateEntropy procedure on sorted file
		 *   > cd ../code
		 *   > java -Xmx1g -classpath . data/ClickEntropy calculateEntropy ../clickentropy/sorted_queries.txt > ../clickentropy/query_entropies.txt &
		 *   
		 * 5) load entropy data into database (note that queries with click-entropy 0.0 are not contained in file)
		 *   mysql> CREATE TABLE aol_click_entropy (Query varchar(600), Entropy DOUBLE, PRIMARY KEY (Query));
		 *   mysql> LOAD DATA LOCAL INFILE 'query_entropies.txt' INTO TABLE aol_click_entropy;  
		 *   
		 */
		
		if (args[0].equals("stemAndSort"))
			stemAndSortQueries(args[1]);
		if (args[0].equals("calculateEntropy"))
			calculateEntropy(args[1]);
		if (args[0].equals("insertEntropy"))
			insertEntropy(args[1]);
		
		
	}
	
	public static void stemAndSortQueries(String filename) {
		
		// tokenizer settings
		boolean removeStopwords = false;
		boolean doStemming = true;
		boolean removePunctuation = true;
		QueryTokenizer qt = new QueryTokenizer(removeStopwords, doStemming, removePunctuation);
								
		try {
				
			Scanner s = new Scanner(new File(filename));
			String line = s.nextLine(); // throw away first line -- it's column headings
			while (s.hasNextLine()) {
				line = s.nextLine(); 
				int tabIndex = line.indexOf('\t');
				String query = line.substring(0, tabIndex).trim();
				String url = line.substring(tabIndex+1);
				// parse query
				ArrayList<String> queryTokens = qt.tokenize(query);
				// sort query
				Collections.sort(queryTokens);
				// write out query
				System.out.println(listToString(queryTokens," ") + "\t" + url);
			}
			
		} 
		catch (Exception e) {e.printStackTrace();}

	}

	
	public static String listToString(ArrayList<String> list, String delimiter) {
		StringBuffer b = new StringBuffer();
		boolean first = true;
		for (String t: list) {
			if (first) first = false;
			else b.append(delimiter);
			b.append(t);
		}	
		return b.toString();
	}
	

	public static String listToString(String[] list, String delimiter) {
		StringBuffer b = new StringBuffer();
		boolean first = true;
		for (String t: list) {
			if (first) first = false;
			else b.append(delimiter);
			b.append(t);
		}	
		return b.toString();
	}
	

	public static void calculateEntropy(String filename) {
		
		try {
				
			Scanner s = new Scanner(new File(filename));
			
			// read first line
			String line = s.nextLine(); 
			int tabIndex = line.indexOf('\t');
			String query = line.substring(0, tabIndex);
			String url = line.substring(tabIndex+1);
			int count = 1;
			
			ArrayList<Integer> counts = new ArrayList<Integer>();
			
			while (s.hasNextLine()) {
				// save previous values
				String previousQuery = query;
				String previousURL = url;
				
				// read next line
				line = s.nextLine(); 
				tabIndex = line.indexOf('\t');
				query = line.substring(0, tabIndex);
				url = line.substring(tabIndex+1);
				
				if (query.equals(previousQuery)) {
					if (url.equals(previousURL)) count++;
					else {	
						counts.add(count);
						count=1;						
					}
				}
				else {
					counts.add(count);
					// print entropy if it's greater than 0
					if (counts.size()>1) System.out.println(query + "\t" + entropy(counts)); 
					counts = new ArrayList<Integer>();
					count=1;
				}
			}

			
		} 
		catch (Exception e) {e.printStackTrace();}

	}
	
	
	public static double entropy(ArrayList<Integer> counts) {
		int total = 0;
		for (int c: counts) total += c;
		double t = total;
		double e = 0;
		for (int c: counts) { 
			double p = c/t;
			e -= p*Math.log(p);
		}
		return e / Math.log(2.0); // return entropy in bits (not nats)
	}
	
	
	public static void insertEntropy(String filename) {
		
		// JDBC connection parameters and driver
		String jdbcURL = "jdbc:mysql://localhost/mark?user=mark&password=taipei101";
		String jdbcDriver = "com.mysql.jdbc.Driver";
		java.sql.Connection con;
				
		try {
				
			// connect to the database
			Class.forName(jdbcDriver).newInstance();
			con = java.sql.DriverManager.getConnection(jdbcURL);
			
			String q1 = "SELECT Entropy FROM aol_click_entropy WHERE Query = ?;";
			java.sql.PreparedStatement ps1 = con.prepareStatement(q1);
			
			Scanner s = new Scanner(new File(filename));
			
			String line = s.nextLine();
			int openBracket = line.indexOf('[');
			System.out.println(line.substring(0, openBracket) + "clickEntropy " + line.substring(openBracket));
			
			while (s.hasNextLine()) {
				line = s.nextLine();
				openBracket = line.indexOf('[');
				String[] queryTokens = line.substring(openBracket+1,line.indexOf(']')).split(",");
				Arrays.sort(queryTokens); // sort tokens
				ps1.setString(1, listToString(queryTokens," "));
				java.sql.ResultSet rs1 = ps1.executeQuery();
				double entropy;
				if (rs1.next())
					entropy = rs1.getDouble(1);					
				else
					entropy = 0.0;
				System.out.printf("%s%g %s\n", line.substring(0, openBracket), entropy, line.substring(openBracket));
			}
			ps1.close();
			
		} 
		catch (Exception e) {e.printStackTrace();}

	}
    

	
}
