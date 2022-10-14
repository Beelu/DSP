#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "../inc/hmm.h"

using namespace std;

int main()
{
	//�إߤ@��map�����^���Ʀr�A���ᰵindex�ϥ� 
	map<char, int> dict;
	dict['A']=0; dict['B']=1; dict['C']=2; dict['D']=3; dict['E']=4; dict['F']=5;

	HMM hmms[5];
	load_models( "../modellist.txt", hmms, 5);
	dump_models( hmms, 5);
	
	int T=50;
	int N=6;
	float correct_num=0;
	
	ifstream infile("../data/test_seq.txt");
	ifstream label_file("../data/test_lbl.txt");
	string test_line;
	string label_line;
	ofstream outfile ("result.txt");
	while(getline(infile, test_line)){
		double total_max[5]={0};		//�ΨӦs����model�̤j���|���v�Ȥ�j�p�A�P�_��sequence�ݩ����model
		double model_max=0;			//�ΨӦs����model���̤j���� 
		string pred_model = "";
		
		for(int q=0; q<5; q++){
			//�}�l�p���U�I�����v�x�}Delta
			double delta[T][N];	
			int idx = 0;
			//��l��
			for(int i=0; i<N; i++){
				idx = dict[test_line[0]];
				delta[0][i] = hmms[q].initial[0] * hmms[q].observation[i][idx];
			}	
			// DP delta
			for(int i=1; i<T; i++){			//���Y�[���� 
				for(int j=0; j<N;j++){		//�b�Y���A 
					//�q�W�@��col�D�̤j���v��
					double max = 0;
					for(int k=0; k<N; k++){
						if(delta[i-1][k]*hmms[q].transition[k][j] > max)
							max = delta[i-1][k]*hmms[q].transition[k][j];
					}
					
					idx = dict[test_line[i]];
					delta[i][j] = max * hmms[q].observation[j][idx];
				}
			}
			
			//�M��D�X�̫�@��col���̤j�ȡA����5��model��j�p
			for(int i=0; i<N; i++){
				if(delta[T-1][i] > total_max[q]) total_max[q] = delta[T-1][i];
			}
			
			//�M��������model���Ӿ��v�̤j 
			if(total_max[q] > model_max){
				model_max = total_max[q];
				
				int w = q+1;
				char int2char = w + '0';
				string read_file_name = "model_0";
				read_file_name += int2char;
				read_file_name += ".txt";
				pred_model = read_file_name;
			}
		}
		
		getline(label_file, label_line);
		if(pred_model.compare(label_line) == 0) correct_num++;
//		outfile << pred_model << " " << model_max << endl;
	}
	cout << "accurracy = " << correct_num/2500 << endl;
	
	infile.close();
	outfile.close();
	label_file.close();
	
	return 0;
}
