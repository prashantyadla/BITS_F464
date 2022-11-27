/*
Same as Face Recognizer except for the change in number of 
output units
*/



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// v prefix is for validation data

char person[100][50],vperson[100][50];            

// in_data is for storing input pixels of input images
double in_data[100][1000],vin_data[100][1000];

// out is 1 output unit hidden and target for hidden and target layers
double out[1],hidden[21],target[1];

// whi is weight from input unit i to hidden unit j
// delta_whip is previous input to hidden delta values
// similarly for output units
double whi[1000][21],delta_whip[1000][21];
double woh[21][1],delta_wohp[21][1];

double delta_h[21],delta_o[1];
double delta_whi[1000][21],delta_woh[21][1];


// lrate = learning rate
double lrate = 0.3,momentum = 0.3;

// counter = count of training examples seen
int counter,vcounter;

// sigmoid function
double sigmoid(double s)           
{
	double e = exp((-1)*s);
	double val = 1/(1+e);
	return val;
}


// randomize function in order to randomize initial weight values
void randomize(int counter)
{
	
	int i,j;
	srand(102194);
	for(i=0;i<1000;i++)
	{
		for(j=0;j<21;j++)
		{
			whi[i][j]=((((double)(rand()%100))/250)-0.2);
		}
	}
	for(i=0;i<21;i++)
	{
		for(j=0;j<1;j++)
		{
			woh[i][j]=((((double)(rand()%100))/250)-0.2);
		}
	}
}



// backpropogation algorithm implementation to learn weights
void backprop(int counter)
{
	int i,j,k,l,m,n,epochs,sum,pos=0;
	epochs =650;
	
	while(epochs--)
	{
		//printf("%d\n",epochs);
		
		for(i=0;i<counter;i++)          // training over each example
		{
			for(j=0;j<20;j++)       // feed-forward to calculate hidden unit activations 
			{
				hidden[j] = 0;
				for(k=0;k<=960;k++)
				{
					hidden[j] += in_data[i][k]*whi[k][j];
				}
				
				hidden[j] = sigmoid(hidden[j]);
			}
			
			hidden[20] = 1;    // bias unit
			for(j=0;j<1;j++)    // feed-forward to calculate output unit activations
			{
				out[j]=0;
				for(k=0;k<=20;k++)
				{
					out[j] += hidden[k]*woh[k][j];
				}
				out[j] = sigmoid(out[j]);
			}
			
			if(strcmp(person[i],"open")==0) // assigning target values for a training example 
				target[0] = 0.1;
			else target[0] = 0.9;
			
			if(epochs==3)     // epochs=3 to know training examples matched when almost training is done
			{
				double max_val=0;
				int max_ind;
				if(target[0]==0.1 && out[0]<0.5)
					pos++;
				else if(target[0]==0.9 && out[0]>0.5)
					pos++;
				if(i==69)
					printf("%d\n",pos);
				//printf("%d ",i);
				//printf("%lf %lf\n",out[0],target[0]);
			}
			// calculating delta values for output layer
			for(j=0;j<1;j++)
			{
				delta_o[j] = (out[j])*(1-out[j])*(target[j]-out[j]);
			}
			
			// calculating delta values for hidden layer
			for(j=0;j<=20;j++)
			{
				sum = 0;
				for(k=0;k<1;k++)
				{
					sum += woh[j][k]*delta_o[k];
				}
				delta_h[j] = (hidden[j])*(1-hidden[j])*sum;
			}
			
			// calculate delta values for weights
			for(j=0;j<=960;j++)
			{
				for(k=0;k<20;k++)
				{
					delta_whi[j][k] = (lrate*(delta_h[k])*(in_data[i][j])) + (momentum*delta_whip[j][k]);
				}
			}
			for(j=0;j<=20;j++)
			{
				for(k=0;k<1;k++)
				{
					delta_woh[j][k] = lrate*(delta_o[k])*(hidden[j]) + (momentum*delta_wohp[j][k]);
				}
			}
			
			
			// updating weights
			for(j=0;j<=960;j++)
			{
				for(k=0;k<20;k++)
				{
					whi[j][k] += delta_whi[j][k];
				}
			}
			for(j=0;j<=20;j++)
			{
				for(k=0;k<1;k++)
				{
					woh[j][k] += delta_woh[j][k];
				}

			}
			
			
			// assigning current delta in weights to previous delta in weight values
			for(j=0;j<=960;j++)
			{
				for(k=0;k<20;k++)
				{
					delta_whip[j][k] = delta_whi[j][k];
				}
			}
			for(j=0;j<=20;j++)
			{
				for(k=0;k<1;k++)
				{
					delta_wohp[j][k] = delta_woh[j][k];
				}
			}
		}
	}
	
}


// extracting pixel values from the images
void in_dataExtractor(char* imgfile,int flag,char *s4)
{
	char line[600],a[10],ch;
	
	int nr,nc,maxval,i,j,k,l,n,f,val,c;
	int m;
	unsigned char uc,byt[1000];
	
	//if(flag==0)
		//printf("%s\n",imgfile);
	FILE *fp;
	fp = fopen(imgfile,"r");
	if(fp==NULL) printf("***\n");
	if(fp!=NULL){
	
	//printf("%ld\n",ftell(fp));
	
	fgets(line,599,fp);
	sscanf(line,"P%d",&val);
	//printf("%d\n",val);
	if (val != 5) 
	{
    	printf("Only handles pgm files (type P5)\n");
    	fclose(fp);
    	return;
    }
	fgets(line, 599, fp);
  	sscanf(line, "%d %d", &nc, &nr);
  	//printf("%d %d\n",nr,nc);
  	//printf("%ld\n",ftell(fp));
  	fgets(line, 599, fp);
  	sscanf(line, "%d", &maxval);
  	//printf("%d\n",maxval);
  	if (maxval > 255) 
  	{
    	printf("Only handles pgm files of 8 bits or less\n");
    	fclose(fp);
    	return;
    }
    
    if(val==5)
    {
    //	printf("%ld\n",ftell(fp));
     	fread(byt,1,960,fp); 
     	
     	// entering pixel data into in_data while training 
     	if(flag==1)
     	{
     		for(i=0;i<960;i++)
    			in_data[counter][i]=(double)((unsigned char)byt[i]);
    		strcpy(person[counter],s4);
    	}
    	else if(flag==0)     // entering pixel data into vin_data while validation & testing
    	{
    		for(i=0;i<960;i++)
    			vin_data[vcounter][i]=(double)((unsigned char)byt[i]);
    		strcpy(vperson[vcounter],s4);
    	}
    	/*if(counter==0)
    		{
    		
    			FILE* han = fopen("some.pgm","w");
    			fprintf(han,"P5\n32 30\n151\n");
    			for(i=0;i<960;i++)
    			{
    				fprintf(han,"%c",byt[i]);
    			}
    			fclose(han);
    		}*/
       	
    }
    
    
    	
  }
	fclose(fp);
    return;
}

void slashFinder(char img[],int l,int flag)
{
	int k,j,i=l-1,c=0,d;
	char imgfile[300],ext[100],s1[50],s2[50],s3[50],s4[50],s5[50];
	while(1)
	{
			if(img[i]=='/')
				{
					c++;
					if(c==1)
					{
						k=i+1;j=0;
						while(k!=l-1)
							ext[j++]=img[k++];
						ext[j]='\0';
						sscanf(ext,"%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]",s1,s2,s3,s4,&d,s5);
						
					}
					if(c==3)
					{
						j=0;
						i++;
						while(i!=l-1)
							imgfile[j++]=img[i++];
						
						imgfile[j]='\0';
						in_dataExtractor(imgfile,flag,s4);// flag=1 while training and flag=0
						                                        // while testing & validating
						break;
					}
				}
				i--;
	}
}

// Extracting output values while validation & testing using weights learned while training
void checker(char *arg,int pole) 
{
        // v prefix for validation & testing
	char img[300];
	FILE *fp;
	int l,i,j,k,pos;	
	
	if((fp=fopen(arg,"r"))==NULL)
	{
		printf("Not able to open validation data set\n");
		exit(-1);
	}
	vcounter=0;
	while(1)
	{
		if(fgets(img,299,fp)!=NULL)
		{
			l = strlen(img);
			slashFinder(img,l,0);
			vcounter++;
		}
		else break;
	}
	pos=0;
	for(i=0;i<vcounter;i++)
	{
		for(j=0;j<970;j++)
		{
			vin_data[i][j] = vin_data[i][j]/256;    // normalizing pixel values
		}
	}
	
	for(i=0;i<vcounter;i++)    // iterating over each test
	{
		for(j=0;j<20;j++)
		{
			hidden[j] = 0;
			for(k=0;k<=960;k++)
			{
				hidden[j] += vin_data[i][k]*whi[k][j];  // feedforward to calculate hidden values 
			}
				
			hidden[j] = sigmoid(hidden[j]);
		}
			
		hidden[20] = 1;
		for(j=0;j<1;j++)
		{
			out[j]=0;
			for(k=0;k<=20;k++)
			{
				out[j] += hidden[k]*woh[k][j];    // feedforward to calculate output values
			}
			out[j] = sigmoid(out[j]);
		}
		
		if(strcmp(vperson[i],"open")==0)
		target[0] = 0.1;
		else target[0] = 0.9;
			
		//printf("%d ",i);
		//printf("%lf %lf\n",out[0],target[0]);
			
			
		// test considered positive if output unit value for that particular person is nearer to target value
		double max_val=0;
		int max_ind;
		if(target[0]==0.1 && out[0]<0.5)
			pos++;
		else if(target[0]==0.9 && out[0]>0.5)   
			pos++;    // pos = count of positive matches
	}
	double vaccuracy;
	if(pole==1)
	{
		printf("%d/%d\n",pos,vcounter);
		vaccuracy = (double)pos/vcounter;    // reporting accuracy values for validation & testing
		printf("%lf\n",vaccuracy);
	}
	return;
}

int main(int argc,char *argv[])
{
	char img[300];
	int i,j,k,l,c,d,pos;
	double vaccuracy;
	if(argc<3)
	{
		printf("File lists not attached\n");
		exit(-1);
	}
	FILE *fp;
	if((fp = fopen(argv[1],"r"))==NULL)
	{
		printf("Not able to open %s",argv[1]);
		exit(-1);
	}
	counter=0;
	while(1)
	{
		if(fgets(img,299,fp)!=NULL)
		{
			l = strlen(img);
			slashFinder(img,l,1);        // extracting pixel data & target values
			counter++;
		}
		else break;
	}
	for(i=0;i<1000;i++)
		for(j=0;j<21;j++)
			delta_whip[i][j]=0;   // setting previous delta values to 0
			
	for(i=0;i<21;i++)
		for(j=0;j<1;j++)
			delta_wohp[i][j]=0;
			
	randomize(counter);   // setting initial weights to random values
	
	for(i=0;i<counter;i++)
	{
		for(j=0;j<970;j++)
		{
			in_data[i][j] = in_data[i][j]/256;  // normalizing pixel values
		}
	}
	
	for(i=0;i<counter;i++)
		in_data[i][960] = 1;    // bias value initialize
		
	backprop(counter);      // backpropogation to learn weights
	
	checker(argv[2],0);               // validation
	if(strcmp(argv[3],"no")!=0)    // doesn't test 
		checker(argv[3],1);       // testing data
	return 0;
}
