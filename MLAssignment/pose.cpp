#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>

#include "DataRecog.cpp"

using namespace std;

class pose
{
public:
    Data D;
    double first_delta[960][6],second_delta[6][4],first[960][6],first_tho[6];
    double second[6][4];
    double error,err;
    double momentum;
    double eta;
    double gamma;
    double out1[960],out2[6],O[4];
    double *T;
    double *mat;
    double delta;
    int lim, lim2;
    void sigmoid_1()   // evaluates sigmoid value.
    {
        double sum;
        int i,k=0;
        while(k<960)
        {
            out1[k]=mat[k];
            k++;
        }

    }
    void sigmoid_2()
    {
        double sum=0.00;
        int i,k=0;
        while(k<6)
        {
            sum=0.00;
            for(i=0;i<960;i++)
            {
                sum+=first[i][k]*out1[i];
                // printf("%0.20lf %0.20lf %0.2lf\n",first[i][k],out1[i],sum);
            
            }
            out2[k]=1/(1+exp(-sum));
            
            k++;
        }
    }
    void sigmoid_3()
    {
        double sum=0.00;
        int i,k=0;
        while(k<4)
        {
            sum=0.00;
            for(i=0;i<6;i++)
            {
                sum+=second[i][k]*out2[i];
            }
            O[k]=1/(1+exp(-sum));
            k++;
        }

    }
    void delta_2() //calculating delta values of output layer
    {
        int i=0,k=0;
        while(i<6)
        {
            k=0;
            while(k<4)
            {
                second_delta[i][k]=eta*(-((T[k]-O[k])*O[k]*(1-O[k])*out2[i])+2*gamma*second[i][k]) + momentum*second_delta[i][k];
                k++;
            }
            i++;
        }
    }
    void delta_1()  //Calculating delta values of hidden layer
    {
        double sum=0.00;
        int j,i=0,k=0;
        while(k<6)
        {
           sum=0.00;
           for(j=0;j<4;j++)
            {
                sum-=O[j]*(1-O[j])*(T[j]-O[j])*second[k][j];
            }
            first_tho[k]=out2[k]*(1-out2[k])*sum;
            k++;
        }
        while(i<960)  // input to hidden layer weight value changes.
        {
            k=0;
            while(k<6)
            {
                first_delta[i][k]=eta*(first_tho[k]*out1[i]+2*gamma*first[i][k])+momentum*first_delta[i][k];
                k++;
            }
            i++;
        }

    }
    void update_2()  // updating hidden to output layer weights.
    {
        int i,j;
        for(i=0;i<6;i++)
        {
            for(j=0;j<4;j++)
            second[i][j]=second[i][j]+second_delta[i][j];
        }
    } 
    void update_1()  // updating input to hidden layer weights
    {
        int i,j;
         for(i=0;i<960;i++)
        {
            for(j=0;j<6;j++)
            first[i][j]=first[i][j]+first_delta[i][j];
        }
    }
    pose()
    {
        int i,j;
        D.Init(3);
        momentum=0.3;
        eta=-0.3;
        gamma=1/(exp(18));
        error=1;
        for(i=0;i<960;i++)
        {
            for(j=0;j<6;j++)
            {
                first[i][j]=(0.5+(((rand())%1000)*1.0)/1000) -1; //randomizing input to hidden layer weights.
                first_delta[i][j]=0;
            }
        }
        for(i=0;i<6;i++)
        {
            for(j=0;j<4;j++)
            {
                second[i][j]=(0.5+(((rand())%1000)*1.0)/1000) -1; //randomizing hidden layer to output weights.
                second_delta[i][j]=0;
            }
            first_tho[i]=0;
        }
        lim = D.loadTrainingData();
        lim2 = D.loadValidationData(0);

    }

    void train()
    {
        int k=0;
        int j=0;
        while(j<50)
        {
            k=0;
            while(k<lim)   //iterating over all the training list images
            {
                pair<double*,double*> p = D.getTrainingData();
                mat=p.first;
                T=p.second;
                sigmoid_1();
                sigmoid_2();
                sigmoid_3();
                delta_2();
                delta_1();
                update_2();
                update_1();
                k++;
            }
            j++;
        }

        
    }
    void validateFinal()
    { 
        lim2 = D.loadValidationData(1);
        validate();
    }
    void validate()
    {
        int k=0;
        error=0;
        while(k<lim2)
        {
            int i=0;
            int max_pos;
            pair<double*,double*> p = D.getValidationData();
            mat=p.first;
            T=p.second;
            sigmoid_1();
            sigmoid_2();
            sigmoid_3();
            double max=O[0];
            for(i=0;i<4;i++)  // getting unit with maximum output value
            {
                if(O[i]>=max)
                {
                    max=O[i];
                    max_pos=i;
                }
            }
            if(T[max_pos]<0.6)   
            {
                error++;
            }
            k++;
        }
        error=error/lim2;
    }
};


double first[960][20],second[20][20];

void print(char *file, int l,int m, int n, bool sec,double error) //printing to console accuracy after number of iterations
{
    FILE *f = fopen(file,"w");
    int i,j;
    fprintf(f, "Accuracy %lf%%\n",(1-error)*100);

    fprintf(f, "Weights in first level\n");
    for(i=0;i<l;i++)
    {
        for(j=0;j<m;j++)
        {
            fprintf(f, "%lf,",first[i][j] );   			//weights in first level.
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");

    if(sec)
    {
        fprintf(f,"Weights in second level\n");
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            {
                fprintf(f,"%lf,",first[i][j] );
            }
            fprintf(f, "\n");
        }
    }
}

int main()
{
    pose p;
    int i,j;
    double perror,error;

    perror = 9999;
    error = 0;

    p.validate();
    error = p.error;
    printf("Initial accuracy %lf%%\n",(1-error)*100);  //printing the initial accuracy.

    for(int z=1;true;z++)
    {
        p.train();
        p.validate();
        error = p.error;
        printf("Accuracy after %d iterations %lf%%\n",z*50*277,(1-error)*100);
        if(perror < error)
        {
            for(i=0;i<960;i++)
                for(j=0;j<6;j++)
                    p.first[i][j]=first[i][j];
            for(i=0;i<6;i++)
                for(j=0;j<4;j++)
                    p.second[i][j]=second[i][j];
            p.validateFinal();
            error = p.error;
            printf("Accuracy on final validation set %lf%%",(1-error)*100);
            char fl[] = "pose.csv";
            print(fl,960,6,4,true,error);
            break;
        }
        for(i=0;i<960;i++)
            for(j=0;j<6;j++)
                first[i][j]=p.first[i][j];  // Assigning back the changed input to hidden layer weights.
        for(i=0;i<6;i++)
        {
            for(j=0;j<4;j++)
            {
                second[i][j]=p.second[i][j];  //Assigning back the changed hidden to output layer weights.
            }
        }
        perror = error;
    }
}
