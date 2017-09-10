% this is demo code comes from my AAAI 2017 paper. I will upload the whole package for the journal version
function [W,history]=admm_shoe(X,Y,M,alpha,beta,rho,label_size)
    
%W=inv(X'*X+beta*eye(size(X,2)))*X'*Y;
W=zeros(size(X,2),size(Y,2));
%V=W*M;
V=zeros(size(W*M));
u=zeros(size(X,2),size(M,2));


[nInstance,nFeat]=size(X);

MAXITER =100;


iter = 1;
groups=label_size;
nTags=size(Y,2);
alpha_ij=alpha;

%U_2=data.U2;
%Lambda_2=data.Lambda2;

while(iter <= MAXITER)
    
        fprintf('ITERATION %d\n',iter);
        tic;
        %%%% update W
        [U_1,Lambda_1]=eig(X'*X);
        [U_2,Lambda_2]=eig(rho/2*M*M'+beta*eye(size(M,1)));
        Q=U_1'*(X'*Y+rho/2*(V+u./rho)*M')*U_2;
        for i=1:size(Q,1)
            for j=1:size(Q,2)
                W_tild(i,j)=Q(i,j)/(Lambda_1(i,i)+Lambda_2(j,j));
            end
        end
        W=U_1*W_tild*U_2';
        time=toc;
        fprintf('Time taken for W Update %f \n',time);
      
        old_V=V;
        Z=W*M-u./rho;        
        %%% update V
        tic;
        for i=1:nFeat
            for k=1:groups
                index_start=nTags*(k-1)+1;
                index_end=nTags*k;
                group_V{k}=updateV(alpha_ij,rho,Z(i,index_start:index_end));
            end
            V(i,:)=cell2mat(group_V(1:groups));
        end
      time=toc;
      fprintf('Time taken for V Update %f \n',time);
   %   assert(~isequal(V,old_V));
      
      u = u + rho*(V-W*M);
      iter = iter + 1;

     r = V-W*M;
     s = rho*(V-old_V)*M'; 
     history.obj(iter) = calcObj(X,Y,V,W,M,beta,alpha_ij,rho,u,nFeat,nTags,groups);
     if iter==MAXITER
         history.W{iter}=W;
         history.V{iter}=V;
         history.u{iter}=u;
     end
     fprintf('the objective function error is %f \n', history.obj(iter));
     
     rho=min(rho*1.1,1e9);
     
     if iter>2
         if history.obj(iter-1)-history.obj(iter)<0.0001
             break;
         end
     end
     
     
     
%      eps_primal =  RELTOL*max([norm(W*M,'fro'),norm(V,'fro')]);
%      eps_dual =  RELTOL*max([norm(W*M,'fro'),norm(V,'fro')]); 
%         
% %      
%     if norm(r) < eps_primal && norm(s) < eps_dual
%         fprintf('FINISHED\n');
%         break
%     elseif norm(r) > SCALETOL*norm(s,'fro')
%         rho=INC_FACTOR*rho;
%         fprintf('Rho changed to %f\n',rho);
%     elseif norm(s) > SCALETOL*norm(r,'fro')
%         rho=rho/DEC_FACTOR;
%         fprintf('Rho changed to %f\n',rho);
%     end
%      
     
end
        
        
        
function obj=calcObj(X,Y,V,W,M,beta,alpha_ij,rho,u,nFeat,nTags,nGroup)

loss=norm(X*W-Y,'fro')^2+beta*norm(W,'fro')^2+rho/2*norm((V-W*M),'fro')^2+trace(u'*(V-W*M));

tmp=0;
for i=nFeat
    for j=1:nGroup
        tmp=tmp+alpha_ij*norm(V(i,(nTags*(j-1)+1):(nTags*j)));
    end
end

obj=loss+tmp;
        

function v=updateV(alpha_ij,rho,Z)

if norm(Z) <= alpha_ij/rho
        v=zeros(size(Z));
else
        v=Z*(norm(Z) - (alpha_ij/rho))/norm(Z); 
end

