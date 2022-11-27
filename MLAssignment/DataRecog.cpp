#include <cstdio>
#include <cstring>
#include <map>

using namespace std;

class Data
{
public:
	double inp[300][960]; // store pixel values of images while training
	double out[300][20]; // store target values of images while training
	double vinp[300][960];  // store pixel values of images while validation
	double vout[300][20];   // store target values of images while validation
	char train[100], valid1[100], valid2[100];
	int prob;

	int dataSize,dataSize1,dp,dp1;
	map<string,int> m; // map in which we assgn an integer to each person

	void Init(int p)
	{
		int i;
		prob =p;
		if(p==3) // p inorder to get correct files
		{
			sprintf(train,"lists/all_train.list");
			sprintf(valid1,"lists/all_test1.list");
			sprintf(valid2,"lists/all_test2.list");
		}
		
		string s[20] ={"an2i","at33","boland","bpm","ch4f","cheyer","choon","danieln","glickman","karyadi","kawamura","kk49","megak","mitchell","night","phoebe""saavik","steffi","sz24","tammo"};
		for(i=0;i<20;i++)
			m[s[i]]=i;
	}

	void readImage(char* src, unsigned char* buf, double *d)
	{
		int i;
		char p;
		FILE *f;
		f = fopen(src,"r"); // reading .pgm files (images)
		// printf("%s",src);
		// fflush(stdout);
		fread(buf,1,13,f);   //read first 13 bytes which are about rows,columns,maxval.
		fread(buf,1,960,f); // read pixel values.
		fclose(f);

		for(i=0;i<960;i++)
			d[i]=((double)buf[i])/156; //pixel value normalisation.
		
		// for(i=0;i<960;i++)
		// 	printf("%lf ",d[i]);
		// printf("\n\n");
	}

	void readOutput(char *src, double *d)
	{
		int j;
		if(prob==3)    //for pose recognition
		{
			for(j=0;j<4;j++)
				d[j]=0.1;
			switch(*(strchr(src,'_')+1))
			{
				case 's':
					d[0] = 0.9;  //straight
					break;
				case 'u':
					d[1] = 0.9;	//up
					break;
				case 'l':
					d[2] = 0.9;	//left
					break;
				case 'r':
					d[3] = 0.9;  	//right
			}
		}
		

	}

	int loadTrainingData()   //To load training data.
	{
		int i,j;
		unsigned char buf[1000];
		FILE *f;
		char src[200];

		f = fopen(train,"r");
		
		for(i=0;true;i++)
		{
			fgets(src,200,f);
			if(feof(f))
				break;
			src[strlen(src)-1]=0;
			readImage(src,buf,inp[i]);
			readOutput(src,out[i]);
		}
		fclose(f);
		dp = 0;
		dataSize = i;
		
		return i;

	}

	int loadValidationData(int st)    //To load valitation data.
	{
		dp1 = 0;
		int i,j;
		unsigned char buf[1000];
		char src[200];
		FILE *f;

		if(st==0)
			f = fopen(valid1,"r");
		else
			f = fopen(valid2,"r");

		for(i=0;true;i++)  
		{
			fgets(src,200,f);
			if(feof(f))
				break;
			src[strlen(src)-1]=0;

			readImage(src,buf,vinp[i]);
			readOutput(src,vout[i]);
		}
		fclose(f);

		dataSize1 = i;    //Maintain count of total images in the file
		return i;
	}

	pair<double *,double *> getTrainingData() //pair inorder to get training data paired with target data
	{
		pair<double *,double*> p = make_pair(inp[dp],out[dp]);
		dp = (dp+1)%dataSize;
		return p;
	}

	pair<double *,double *> getValidationData()   //pair inorder to get validation data paired with target data
	{
		pair<double *,double*> p = make_pair(vinp[dp1],vout[dp1]);
		dp1 = (dp1+1)%dataSize1;
		return p;
	}
};
