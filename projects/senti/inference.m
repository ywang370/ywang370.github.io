function [U_0,U_0_baseline,U,V,err,junk]=inference1(X,V_0,label,beta,gamma)
 %%% multi-view sentiment analysis
%   This function solves min sum|X-UV'|_F+\alpha|V-V_0|21+\beta|U-U_0|_F
%   [U0,U,V,err]=inference(rand(100,100,3),rand(100,3,3),3,0.5,0.5);

iter=200;
m=size(X{1},1);
k=size(label,2);

alpha=0.5;


num_view=size(X,2); 
num_sentiment=size(label,2);

% other initialization could be better.
for i=1:num_view
    U{i}=rand(size(X{i},1),num_sentiment);
    V{i}=rand(size(X{i},2),num_sentiment);
end


U_0=rand(size(label));

k=num_sentiment;
for j=1:1
    d{j}= 0.5./sqrt(sum((V{j}-V_0{j}).*(V{j}-V_0{j}),2)+eps);
end

for i=1:iter
        obj=0;

        %%% initialize D
        U_t=U{1};        
        X_t=X{1};
        V_t=V{1};
        U_v=U{2};        
        X_v=X{2};
        V_v=V{2};
        D=diag(abs(d{1}));          
        M=U_t'*U_t;
        delta=beta/alpha;
        [evec1,eval1]=eig(M);
        [evec2,eval2]=eig(D);
        Q=evec2'*(X_t'*U_t+delta*D*V_0{1})*evec1;
        Q_1=zeros(size(Q));
           for m=1:size(evec1,1)
              for n=1:size(evec2,2);
                  Q_1(n,m)=delta*eval2(n,n)+eval1(m,m);     
              end
           end
       V_tuda=Q./Q_1;
      V_t=evec2*V_tuda*evec1';
      d{1}= 0.5./sqrt(sum((V_t-V_0{1}).*(V_t-V_0{1}),2)+eps);

      V_v=X_v'*U_v*inv(U_v'*U_v);
      
      vv_p=(abs(V_v'*V_v)+V_v'*V_v)/2;
      vv_n=(abs(V_v'*V_v)-V_v'*V_v)/2;
      xv_p=(abs(X_v*V_v)+X_v*V_v)/2;
      xv_n=(abs(X_v*V_v)-X_v*V_v)/2;     
      U_v=U_v.*sqrt((xv_p+U_v*vv_n+beta*U_0)./(xv_n+U_v*vv_p+beta*U_v+eps));
      
      vv1_p=(abs(V_t'*V_t)+V_t'*V_t)/2;
      vv1_n=(abs(V_t'*V_t)-V_t'*V_t)/2;
      xv1_p=(abs(X_t*V_t)+X_t*V_t)/2;
      xv1_n=(abs(X_t*V_t)-X_t*V_t)/2;     
      U_t=U_t.*sqrt((xv1_p+U_t*vv1_n+delta*U_0)./(xv1_n+U_t*vv1_p+delta*U_t+eps));
      
      U{1}=U_t;
      U{2}=U_v;
      V{1}=V_t;
      V{2}=V_v;
      Lambda=(U_0'*U_v-eye(k)+U_0'*U_t-eye(k));
      UL_p=(abs(U_0*Lambda)+U_0*Lambda)/2;
      UL_n=(abs(U_0*Lambda)-U_0*Lambda)/2;
      U_0=U_0.*sqrt((U_t+U_v+UL_n)./(U_0+U_0+UL_p+eps));
      obj=sum(sum((X_v-U_v*V_v')'*(X_v-U_v*V_v')))+gamma*sum(sum((V_t-V_0{1}).*(V_t-V_0{1})))+beta*sum(sum((U_v-U_0)'*(U_v-U_0)))...
          +alpha*sum(sum((X_t-U_t*V_t')'*(X_t-U_t*V_t')))+beta*sum(sum((U_t-U_0)'*(U_t-U_0)));
      
    
     err(i)=obj;  

 end


figure,plot(err)


