#include <math.h>
#include <iostream>
#include "../inc/hmm.h"
#include <sstream>
#include <string>
#include <fstream>
#include <map>
using namespace std;

int main()
{
	//�إߤ@��map�����^���Ʀr�A���ᰵindex�ϥ� 
	map<char, int> dict;
	dict['A']=0; dict['B']=1; dict['C']=2; dict['D']=3; dict['E']=4; dict['F']=5;
	
	int N = 6;
	int T = 50;
	int num_of_seq = 10000;
	int iters = 1;
		
	//��Ū����l�� 
	HMM hmm_initial;
	loadHMM( &hmm_initial, "../model_init.txt" );
	dumpHMM( stderr, &hmm_initial );

	double temp = 0;
	//��mbi(t)�A������B�z������[²��
	for(int i=0; i<6; i++){
        for(int j=i; j<6; j++){
            temp = hmm_initial.observation[i][j];
            hmm_initial.observation[i][j] = hmm_initial.observation[j][i];
            hmm_initial.observation[j][i] = temp;
        }
    }
    
 	for(int w=1; w<6; w++){
 		char int2char = w + '0';
		string read_file_name = "../data/train_seq_0";
		read_file_name += int2char;
		read_file_name += ".txt";
		
		string write_file_name = "model_0";
		write_file_name += int2char;
		write_file_name += ".txt";
		cout << read_file_name <<" ";
	
	 	for(int q=0; q<iters; q++){						//���X��iterations 
			// �v��Ū��train_seq
			ifstream infile(read_file_name.c_str());
			string train_line;
			//��l�Ƴ̫��s�n�Ϊ����ӭ� 
			double total1[N], total2[N][N], total3[N][N], total4[N][N], total5[N][N];
			for(int i=0; i<N; i++){
				total1[i] = 0;
				for(int j=0; j<N; j++){
					total2[i][j] = 0;
					total3[i][j] = 0;
					total4[i][j] = 0;
					total5[i][j] = 0;
				}
			}
			
			while(getline(infile, train_line)){		//Ū���@�U��sequence 
				//============================================�h�W���p��4�ӭ�===========================================// 
				//�p����k 
				double alpha[T][N];
				// alpha��l�� 
				int idx = dict[train_line[0]];
				for(int i=0; i<N; i++){
					alpha[0][i] = hmm_initial.initial[i] * hmm_initial.observation[i][idx];
				}
				// DP alpha
				for(int i=1; i<T; i++){							//�qo1~o50���[��ǦC 
					idx = dict[train_line[i]];
					for(int j=0; j<N; j++){						//�Y���[���ot�q���A1~���A6 
						double last_state_total = 0;
						for(int k=0; k<N; k++){
							last_state_total += (alpha[i-1][k] * hmm_initial.transition[k][j]);
						}
						alpha[i][j] = last_state_total * hmm_initial.observation[j][idx];
					}
				}
				
				//�p��Beta
				double beta[T][N];
				// beta��l�� 
				for(int i=0; i<N; i++){
					beta[T-1][i] = 1;
				}
				// DP beta
				for(int i=T-1; i>0; i--){			//�qo50~o1���[��ǦC 
					idx = dict[train_line[i]];
					for(int j=0; j<N; j++){			//�Y���[���ot�q���A1~���A6
						double last_state_total = 0;
						for(int k=0; k<N; k++){
							last_state_total += (hmm_initial.transition[j][k] * hmm_initial.observation[k][idx] * beta[i][k]);
						}
						beta[i-1][j] = last_state_total;
					}
				}
				
				//�p��gamma 
				double gamma[T][N];
				for(int i=0; i<T; i++){
					//��������`��
					double this_total = 0;
					for(int j=0; j<N; j++){
						this_total += alpha[i][j] * beta[i][j];
					}
					
					// �A��gamma 
					for(int j=0; j<N; j++){
						gamma[i][j] = (alpha[i][j] * beta[i][j]) / this_total;
					}
				}
				
				//�p��epsilon
				double epsilon[T][N][N];
				for(int i=0; i<T-1; i++){					//���Ҧ����[��ǦCot��ot+1 
					idx = dict[train_line[i+1]];
					//��������`�� 
					double this_total = 0;
					for(int j=0; j<N; j++){
						for(int k=0; k<N; k++){
							this_total += alpha[i][j] * hmm_initial.transition[j][k] * hmm_initial.observation[k][idx] * beta[i+1][k];
						}
					}
					
					//�A��epsilon
					for(int j=0; j<N; j++){				//�q���Aj 
						for(int k=0; k<N; k++){			//���쪬�Ak 
							epsilon[i][j][k] = (alpha[i][j] * hmm_initial.transition[j][k] * hmm_initial.observation[k][idx] * beta[i+1][k]) / this_total;
						}
					}
				}
				
				//==========�[�`==========//
				//pi
				for(int i=0; i<N; i++){
					total1[i] += gamma[1][i];
				}
				
				//aij
				for(int i=0; i<N; i++){				//�q���Ai 
					for(int j=0; j<N; j++){			//�쪬�Aj 
						double total_gamma = 0;
						double total_epsilon = 0;
						for(int k=0; k<T-1; k++){
							total_gamma += gamma[k][i];
							total_epsilon += epsilon[k][i][j];
						}
						total2[i][j] += total_epsilon;
						total3[i][j] += total_gamma;
					}
				}
				
				
				//bij
				for(int i=0; i<N; i++){				//�b���Ai�� 
					for(int j=0; j<N; j++){			//�ݨ��[���Ok�����v 
						double total_gamma_top = 0;
						double total_gamma_down = 0;
						for(int k=0; k<T; k++){
							total_gamma_down += gamma[k][i];
							if(train_line[k]-65 == j){
								total_gamma_top += gamma[k][i];
							}
						}
						total4[i][j] += total_gamma_top;
						total5[i][j] += total_gamma_down;
					} 
				}
			}
				
			//=================================================�h�W�����A�}�l��s========================================================//
		//	//����spi
		//	for(int i=0; i<N; i++){
		//		hmm_initial.initial[i] = gamma[1][i];
		//	}
		//	
		//	//�A��saij
		//	for(int i=0; i<N; i++){				//�q���Ai 
		//		for(int j=0; j<N; j++){			//�쪬�Aj 
		//			double total_gamma = 0;
		//			double total_epsilon = 0;
		//			for(int k=0; k<T-1; k++){
		//				total_gamma += gamma[k][i];
		//				total_epsilon += epsilon[k][i][j];
		//			}
		//			hmm_initial.transition[i][j] = total_epsilon / total_gamma;
		//		}
		//	}
		//	
		//	
		//	//�̫��sbij
		//	for(int i=0; i<N; i++){				//�b���Ai�� 
		//		for(int j=0; j<N; j++){			//�ݨ��[���Ok�����v 
		//			double total_gamma_top = 0;
		//			double total_gamma_down = 0;
		//			for(int k=0; k<T; k++){
		//				idx = dict[train_line[k]];
		//				total_gamma_down += gamma[k][i];
		//				if(idx == j){
		//					total_gamma_top += gamma[k][i];
		//				}
		//			}
		//			hmm_initial.observation[i][j] = total_gamma_top / total_gamma_down;
		//		} 
		//	}
			for(int i=0; i<N; i++){
				hmm_initial.initial[i] = total1[i] / num_of_seq;
			}
			for(int i=0; i<N; i++){
				for(int j=0; j<N; j++){
					hmm_initial.transition[i][j] = total2[i][j] / total3[i][j];
					hmm_initial.observation[i][j] = total4[i][j] / total5[i][j];
				}
			}
			
			infile.close();
		}
		
		//===================================�V�m���}�l�g�ɮ�====================================//
		ofstream outfile (write_file_name.c_str());
	
		outfile << "initial: 6" << endl;
		outfile << hmm_initial.initial[0];
		for(int i=1; i<N; i++){
			outfile << "	" << hmm_initial.initial[i];
		}
		outfile << endl << endl;
		
		outfile << "transition: 6" << endl;
		for(int i=0; i<N; i++){
			outfile << hmm_initial.transition[i][0];
			for(int j=1; j<N; j++){
				outfile << "	" << hmm_initial.transition[i][j];
			}
			outfile << endl;
		}
		outfile << endl;
		
		outfile << "observation: 6" << endl;
		for(int i=0; i<N; i++){
			outfile << hmm_initial.observation[i][0];
			for(int j=1; j<N; j++){
				outfile << "	" << hmm_initial.observation[i][j];
			}
			outfile << endl;
		}
		outfile.close();
	}

	return 0;
}
