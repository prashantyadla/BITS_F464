#include <iostream>
#include <cstdio>
#include<cmath>
using namespace std;

class DataExtraction // This class extracts the pixels and converts them into 1's and 0's and stores them in arrays aandb.
{
  public:
	int** a;    // a is for training data
	int** b;    // b is for test data
	
	DataExtraction() //Constructor used to provide the functionality of class.
	{
		int i,j,k;
		char c[80];
		a = new int*[451];   // a is pointer to 451 training 
		for(i=0;i<451;i++)
			a[i] = new int[5000];   
		b = new int*[451];
		for(i=0;i<150;i++)   //  b is pointer to 150 testing data
			b[i] = new int[5000];
			
		FILE *fp;
		fp = fopen("facedatatrain","r");   // extracting pixels from training data
		for(i=0;i<451;i++)
		{
			for(j=0;j<70;j++)          // rows of each image 70
			{
				fgets(c,65,fp);
				for(k=0;k<60;k++)     // columns of each image  60
				{
					if(c[k]=='#')
					     a[i][j*60+k] = 1;// if we see hash, set that corresponding value in array to 1
					else a[i][j*60+k] = 0; // else 0
				}
			}
		}
		
		                // doing the same for testing data
		fp = fopen("facedatatest","r");
		for(i=0;i<150;i++)
		{
			for(j=0;j<70;j++)
			{
				fgets(c,65,fp);
				for(k=0;k<60;k++)
				{
					if(c[k]=='#')
						b[i][j*60+k] = 1;
					else b[i][j*60+k] = 0;
				}
			}
		}
		fp = fopen("facedatatrainlabels","r");
		for(i=0;i<451;i++)
		{
			fgets(c,10,fp);
			if(c[0]=='1')
			a[i][60*70+1]=1;// storing the actual value(whether face or not) in the last element of array
			else a[i][60*70+1]=0;
		}
		// doing the same for testing data
		fp = fopen("facedatatestlabels","r");
		for(i=0;i<150;i++)
		{
			fgets(c,10,fp);
			if(c[0]=='1')
				b[i][60*70+1]=1;
			else b[i][60*70+1]=0;
		}
		/*for(i=0;i<70;i++)
		{
			for(j=0;j<60;j++)
				cout<<b[0][i*60+j]<<" ";
			cout<<endl;
		}*/
		//for(i=0;i<451;i++)
			//cout<<a[i][4201]<<endl;
	}
};
class Bayes              // implementation of naive bayes algorithm
{
 public:
  double at1_pos[5000];   // storing in each pixel position count of how many times 1 has occured when image is face 
  double at1_neg[5000];   // storing in each pixel position count of how many times 1 has occured when image is not face
  long tot_train,tot_test,pixels;   // total training examples, total testing examples, total pixels in image
  long m;   // equivalent sample size
  double error;  // 
  double P_1,P_0;   // P_0 is count of non-faces, P_1 is count of faces in training data
  int *in;
  int out;
  Bayes()
  {
      tot_train=451;tot_test=150;pixels=4200;m=2;
      for(int i=0;i<pixels;i++)
      {
          at1_pos[i]=0;      // initializing all count of attribute values to 0 given image is face
          at1_neg[i]=0;      // initializing all count of attribute values to 0 given image is not face
      }
  }
  void counter()
  {
      DataExtraction* D = new DataExtraction();   // making use of the extraction class on trianing data
      long i;
      for(i=0;i<tot_train;i++)
      {
          out=D->a[i][4201];   
          in=D->a[i];
          update();    // updating at1_pos and at1_neg arrays values
      }
      
  }
  void Test()
  {
      long i,j;
      double pos,neg;  
      DataExtraction* D = new DataExtraction();
      for(j=0;j<tot_test;j++)
      {
          pos=0;                                        //pos is logarithm of image being positive(face) 
          neg=0;                                        //neg is logarithm of image being negative(not face) 
       //   cout<<"***"<<endl;
          out=D->b[j][4201]; 
          in=D->b[j];
          for(i=0;i<pixels;i++)                         // for each pixel of image
          {
              if(in[i]==1)
              {
                  pos+=log((at1_pos[i]+0.5*m)/(P_1 + m));  // if attribute value is one , pos is added with log(P(ai=1|vj=1))
              }                                             
              else
              {
                  pos+=log((P_1 - at1_pos[i]+m*0.5)/(P_1+m));// if attribute value is one , pos is added with log(P(ai=0|vj=// vj=1))
              }                                                                                                                                                 
          }
          pos+=log(P_1/(P_0 + P_1));  // log(P(vj=1))
          for(i=0;i<pixels;i++)
          {
              if(in[i]==1)
              {
                  neg+=log((at1_neg[i]+0.5*m)/(P_0 + m));// if attribute value is one , pos is added with log(P(ai=1|vj=0))
              }
              else
              {
                  neg+=log((P_0 - at1_neg[i]+m*0.5)/(P_0+m));// if attribute value is one , pos is added with log(P(ai=0|vj=// vj=0))
              }
          }
          neg+=log(P_0/(P_0 + P_1));   // log(P(vj=0))
          if((pos>neg && out==1) || (pos<neg && out==0))
          {
              //no error
              cout<<"pos,neg and out: "<<pos<<" "<<neg<<" "<<out<<endl;
          }
          else
          {
              error+=1.0;
              cout<<"ERROR!!! pos,neg and out: "<<pos<<" "<<neg<<" "<<out<<endl;
          }
    }
	double accuracy = ((tot_test - error)/tot_test)*100;
	cout<<"accuracy is "<<accuracy<<endl;
  }
  void update()
  {
      long i=0;
      if(out==0)
      {
          P_0+=1.0;
          for(i=0;i<pixels;i++)
          {
              if(in[i]==1)
              at1_neg[i]+=1.0;     // when image is not face, and attribute value is one, then updating at1_neg
          }
      }
      else
      {
          P_1+=1.0;
          for(i=0;i<pixels;i++)
          {
              if(in[i]==1)
                at1_pos[i]+=1.0;   // when image is face , and attribute value is one, the updating at1_pos
          }
      }
    }
};
int main()
{
	int i,j,k,l,m,n;
	Bayes* nbobj = new Bayes();
	nbobj->counter();
	nbobj->Test();
}
