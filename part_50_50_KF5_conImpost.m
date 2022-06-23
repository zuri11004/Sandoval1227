%%zurisaddai Sandoval Lara
%Tesis: Self-organizing clustering  by Growing-SOM for EEG Biometrics
%Asesora: Dra. Pilar Gómez Gil
%modificado: 25 de abril 2020.
%particion de datos para k-folds de diferentes conjuntos de datos para la
%evaluacion biométrica del sistema

 %programa para particionar datos deacuerdo a clases balanceadas donde el
%sujeto 1, 3,4,5,6 tendran 10 ensayos por lo cual de usara el 50 porciento para validar y el 50 para entrenar al azar usando particiones de k-flolds, ademas se usara el
%sujeto 2 y 7 como impostores, asi cumplira las condiciones reales de un
%sistema biometrico.
clear all
clc
clear variables;

% leyendo la tarea deseada.

       load('KEIRN_task_rotation'); % se carga la matriz original con sus etiquetas 
       %load('KEIRN_task_multiplication'); % se carga la matriz original con sus etiquetas
       %load('KEIRN_task_letter-composing'); % se carga la matriz original con sus etiquetas 
       %load('KEIRN_task_baseline'); % se carga la matriz original con sus etiquetas 
       %load('KEIRN_task_counting'); % se carga la matriz original con sus etiquetas 

       
       recortado1= recortado;
       [m,n]=size(recortado1);
       i_escogido=1;
       j_escogido=1;
       % se dividira el conjunto 
       T_usuarios = cell(m,50);
       T_impostores=cell(m,10);
       for i=1:n
           if(recortado1{1,i}{1,3}== "trial 1" || recortado1{1,i}{1,3}== "trial 2"|| recortado1{1,i}{1,3}== "trial 3"|| recortado1{1,i}{1,3}== "trial 4"|| recortado1{1,i}{1,3}== "trial 5" || recortado1{1,i}{1,3}== "trial 6" || recortado1{1,i}{1,3}== "trial 7"|| recortado1{1,i}{1,3}== "trial 8"|| recortado1{1,i}{1,3}== "trial 9"|| recortado1{1,i}{1,3}== "trial 10")
               if(recortado1{1,i}{1,1}== "subject 1" ||recortado1{1,i}{1,1}== "subject 3" ||recortado1{1,i}{1,1}== "subject 4" ||recortado1{1,i}{1,1}== "subject 5" ||recortado1{1,i}{1,1}== "subject 6" )
                 T_usuarios{1,i_escogido}=recortado1{1,i};
                 i_escogido = i_escogido+1;
               elseif(recortado1{1,i}{1,1}== "subject 2" ||recortado1{1,i}{1,1}== "subject 7")
                 T_impostores{1,j_escogido}=recortado1{1,i};
                 j_escogido = j_escogido+1;  
               end
           end
       end
      
      
        
        Ind_c = cvpartition(50,'kfold',5);
        %50 es el tamaño de las celdas de usuarios 5 usuarios con 10
        %ensayos el metodo es kfold y 10 es el numero de conjuntos
        fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
        fprintf(' multiplication \n rotation \n counting\n');
        tarea = input('escribe entre comillas simples la tarea  ');
        %Guardar los 10 conjuntos
        nombre=cat(2,'KEIRN_impost_',tarea);
        save(nombre,'T_impostores'); % se guarda el conjunto impostor
        for i=1:5
            if(i==1)
%                 T_train=cell(45,1); %conjunto 1 de entrenamiento
%                 T_test=cell(5,1); %conjunto 1 de testing
%                 con_1=test(Ind_c,i);
%                 idx=1;
%                 jdx=1;           
%                 [Ind_m,~]=size(con_1);
%                 for h=1:Ind_m
%                   if(con_1(h,1)==0)
%                      T_train{idx,1}= T_usuarios{h,1};
%                      idx=idx + 1;      
%                   else
%                      T_test{jdx,1}= T_usuarios{h,1};
%                      jdx=jdx + 1; 
%                   end
%                 end
%               T_train=T_train';
%                T_test=T_test';
%                 
                
                 [T_train_c1, T_test_c1]=conjunto_folds(T_usuarios,i,Ind_c);
                  nombre1=cat(2,'KEIRN_trainingc1_',tarea);
                  save(nombre1,'T_train_c1');
                  nombre2=cat(2,'KEIRN_testingc1_',tarea);
                  save(nombre2,'T_test_c1');
                  fprintf('conjunto 1');
            end
            if(i==2)
                [T_train_c2, T_test_c2]=conjunto_folds(T_usuarios,i,Ind_c);
                  nombre1=cat(2,'KEIRN_trainingc2_',tarea);
                  nombre2=cat(2,'KEIRN_testingc2_',tarea);
                  save(nombre1,'T_train_c2');
                  save(nombre2,'T_test_c2');
                  fprintf('conjunto 2');
            end
            if(i==3)
                [T_train_c3, T_test_c3]=conjunto_folds(T_usuarios,i,Ind_c);
                  nombre1=cat(2,'KEIRN_trainingc3_',tarea);
                  nombre2=cat(2,'KEIRN_testingc3_',tarea);
                  save(nombre1,'T_train_c3');
                  save(nombre2,'T_test_c3');
                  fprintf('conjunto 3');
            end
             if(i==4)
                [T_train_c4, T_test_c4]=conjunto_folds(T_usuarios,i,Ind_c);
                  nombre1=cat(2,'KEIRN_trainingc4_',tarea);
                  nombre2=cat(2,'KEIRN_testingc4_',tarea);
                  save(nombre1,'T_train_c4');
                  save(nombre2,'T_test_c4');
                  fprintf('conjunto 4');            
             end
             if(i==5)
                [T_train_c5, T_test_c5]=conjunto_folds(T_usuarios,i,Ind_c);
                  nombre1=cat(2,'KEIRN_trainingc5_',tarea);
                  nombre2=cat(2,'KEIRN_testingc5_',tarea);
                  save(nombre1,'T_train_c5');
                  save(nombre2,'T_test_c5');
                  fprintf('conjunto 5');
             end
%             if(i==6)
%                 [T_train_c6, T_test_c6]=conjunto_folds(T_usuarios,i,Ind_c);
%                   nombre1=cat(2,'KEIRN_trainingc6_',tarea);
%                   nombre2=cat(2,'KEIRN_testingc6_',tarea);
%                   save(nombre1,'T_train_c6');
%                   save(nombre2,'T_test_c6');
%                   fprintf('conjunto 6');
%             end
%             if(i==7)
%                 [T_train_c7, T_test_c7]=conjunto_folds(T_usuarios,i,Ind_c);
%                   nombre1=cat(2,'KEIRN_trainingc7_',tarea);
%                   nombre2=cat(2,'KEIRN_testingc7_',tarea);
%                   save(nombre1,'T_train_c7');
%                   save(nombre2,'T_test_c7');
%                   fprintf('conjunto 7');
%             end
%             if(i==8)
%                 [T_train_c8, T_test_c8]=conjunto_folds(T_usuarios,i,Ind_c);
%                   nombre1=cat(2,'KEIRN_trainingc8_',tarea);
%                   nombre2=cat(2,'KEIRN_testingc8_',tarea);
%                   save(nombre1,'T_train_c8');
%                   save(nombre2,'T_test_c8');
%                   fprintf('conjunto 8');
%             end
%             if(i==9)
%                 [T_train_c9, T_test_c9]=conjunto_folds(T_usuarios,i,Ind_c);
%                   nombre1=cat(2,'KEIRN_trainingc9_',tarea);
%                   nombre2=cat(2,'KEIRN_testingc9_',tarea);
%                   save(nombre1,'T_train_c9');
%                   save(nombre2,'T_test_c9');
%                   fprintf('conjunto 9');
%             end
%             if(i==10)
%                 [T_train_c10, T_test_c10]=conjunto_folds(T_usuarios,i,Ind_c);
%                   nombre1=cat(2,'KEIRN_trainingc10_',tarea);
%                   nombre2=cat(2,'KEIRN_testingc10_',tarea);
%                   save(nombre1,'T_train_c10');
%                   save(nombre2,'T_test_c10');
%                   fprintf('conjunto 10');
            %end            
        end
        T_usuarios=T_usuarios';
    
      
      
        
function [T_train,T_test]=conjunto_folds(T_usuarios,i,Ind_c) %funcion que genera los conjuntos de prueba
 
              T_train=cell(40,1); %conjunto 1 de entrenamiento
              T_test=cell(10,1); %conjunto 1 de testing
              con_1=test(Ind_c,i);
              idx=1;
              jdx=1;           
              [Ind_m,~]=size(con_1);
              for h=1:Ind_m
                  if(con_1(h,1)==0)
                     T_train{idx,1}= T_usuarios{1,h};
                     idx=idx + 1;      
                  else
                     T_test{jdx,1}= T_usuarios{1,h};
                     jdx=jdx + 1; 
                  end
              end
              T_train=T_train';
               T_test=T_test';
end
 

