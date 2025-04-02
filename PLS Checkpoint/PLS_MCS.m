%A4 Q1 - function

function[t,u,w_star,c,p,R2]=PLS_MCS(X,Y,n)
    Yraw = Y;
    Xraw = X;

    %process data
    Xmean = mean(Xraw);
    Xstd = std(Xraw);
    X = (Xraw-Xmean)./Xstd;

    Ymean = mean(Yraw);
    Ystd = std(Yraw);
    Y = (Yraw-Ymean)./Ystd;

    %initialize outputs
    t = zeros(length(X),n);
    u = zeros(length(X),n);
    w = zeros(width(X),n);
    w_star = zeros(width(X),n);
    c = zeros(width(Y),n);
    p = zeros(width(X),n);
    R2 = ones(1,n);
    
    %start NIPALS PLS
    tol = 10^-10;
    j = ones(1,n);
    max_iters = 300;
    
    %for loop for the number of components
    for i = 1:n
        % 1) select the first col of y arbitrarily to be u
        u(:,i) = Y(:,1); 
            
        %for loop for the NIPALS
        while j(i) <max_iters
            j(i) = j(i)+1;
            %2.1) Regress each col of x on u to get w
            w(:,i) = transpose(u(:,i))*X/(transpose(u(:,i))*u(:,i));
        
            %2.2) Normalize the weights
            w(:,i) = transpose(w(:,i)/norm(w(:,i)));

            %2.3) Regress each row of X onto wT to get t
            t(:,i) = X*w(:,i)/(transpose(w(:,i))*w(:,i));
        
            %2.4) Regress each col of Y onto t to get c
            c_transpose = transpose(t(:,i))*Y/(transpose(t(:,i))*t(:,i));
            c(:,i) = transpose(c_transpose);
        
            %2.5) Regress each row of Y onto c to get u_new
            u_new = Y*(c(:,i))/(transpose(c(:,i))*c(:,i));
        
            %check for convergence 
            change = abs(u_new-u(:,i));
            max_change = max(change);
            u(:,i) = u_new;

            a = abs(max_change);
    
            if a< tol
                
                %Calculate R2 for the component
                y_model_new_mcs = t(:,i)*transpose(c(:,i));
                y_model_new_raw = (y_model_new_mcs*transpose(Ystd))+Ymean;
            
                num = sum(((y_model_new_raw-Yraw).^2),"all");
                dom = sum(((sum((Yraw-Ymean).^2))),"all");
                R2(:,i) = R2(:,i) - num/dom;
                
                if i>1
                    R2(:,i) = R2(:,i)+R2(:,i-1);
                end

                j(i) = max_iters;
            end    
        end

        % 3.1) Compute loadings p by regressing cols of X onto converged t
        p(:,i) = transpose(X)*t(:,i)/(transpose(t(:,i))*t(:,i));

        %3.2) Remove predicted var in X and Y
        x_model = t(:,i)*transpose(p(:,i));
        X = X-x_model;
        
        y_model = t(:,i)*transpose(c(:,i));
        Y = Y-y_model;
    end

    %make weights interpretable
    w_star = w*inv(transpose(p)*w);

end