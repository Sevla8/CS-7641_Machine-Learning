package assignment4;

public class MyMap {

	private int row;
	private int col;
	private double successRate;
	private int[][] coordinates;
	
	public MyMap(int row, int col, double successRate, double wallPercent){
		this.row = row;
		this.col = col;
		this.successRate = successRate;
		coordinates = new int[row][col];
		this.mapSetup(wallPercent);
	}
	
	
	private void mapSetup(double wallPercent) {
		int wallNum = (int) Math.round((row*col)*wallPercent);
		for(int i=0;i<wallNum;i++){
			int ran_row = (int)(Math.random() * row);
			int ran_col = (int)(Math.random() * col);
			while((ran_row==0&&ran_col==0)||coordinates[ran_row][ran_col] == 1||(ran_row==row&&ran_col==col)){
				ran_row = (int)(Math.random() * row);
				ran_col = (int)(Math.random() * col);
			}
			coordinates[ran_row][ran_col] = 1;
		}
		
	}


	public int[][] getCoordinates() {
		return this.coordinates;
	}
	
	public int getRow(){
		return this.row;
	}
	
	public int getCol(){
		return this.col;
	}
	
	public double getSuccessRate(){
		return this.successRate;
	}
	
	
}
