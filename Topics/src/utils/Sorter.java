package utils;

import java.util.ArrayList;
import java.util.Collections;

class IntDouble implements Comparable<IntDouble> {
	int id;
	double val;
	public IntDouble(int i, double d) { id = i; val = d; }
	public int compareTo(IntDouble idp) {
		return - (new Double(val)).compareTo(idp.val);
	}
}

public class Sorter {
	
	
	public static int[] rankedList(double[] vals) {
		ArrayList<IntDouble> list = new ArrayList<IntDouble>();
		int id = 0;
		for (double val: vals) list.add(new IntDouble(id++,val));
		Collections.sort(list);
		int[] ranked = new int[vals.length];
		for (int i=0; i<vals.length; i++) 
			ranked[i] = list.get(i).id;
		return ranked;
	}

}
