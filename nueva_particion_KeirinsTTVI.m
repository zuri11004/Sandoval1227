 %zurisaddai Sandoval Lara
%Tesis: Self-organizing clustering  by Growing-SOM for EEG Biometrics
%Asesora: Dra. Pilar Gómez Gil
%modificado: 29 de febrero del 2020.
%Programa que crea una particion de 3 conjuntos de datos
%base de datos de keirins. 


%programa para particionar datos deacuerdo a clases balanceadas donde el
%sujeto 1, 3,4,5,6 tendran 5 ensayos del dia uno con el que se entrenara y
%5 ensayos del dia dos con el que se probara y validara, ademas se usara el
%sujeto 2 y 7 como impostores, asi cumplira las condiciones reales de un
%sistema biometrico.
clear all;
clear variables;

% leyendo la tarea deseada.

       %load('KEIRN_task_rotation'); % se carga la matriz original con sus etiquetas 
       %load('KEIRN_task_multiplication'); % se carga la matriz original con sus etiquetas
       load('KEIRN_task_letter-composing'); % se carga la matriz original con sus etiquetas 
       %load('KEIRN_task_counting'); % se carga la matriz original con sus etiquetas 
       %load('KEIRN_task_baseline'); % se carga la matriz original con sus etiquetas 
       %matriz_datos= recortado{1,1}{1,4};
       %matriz_canales=matriz_datos(1:6,:); %matriz de todos los 6 canales
       recortado1= recortado;
       [m,n]=size(recortado1);
       i_escogido=1;
       j_escogido=1;
       k_escogido=1;
       Training = cell(m,25);
       Resto=cell(m,25);
       Impostor=cell(m,10);
       for i=1:n
           if(recortado1{1,i}{1,3}== "trial 1" || recortado1{1,i}{1,3}== "trial 2"|| recortado1{1,i}{1,3}== "trial 3"|| recortado1{1,i}{1,3}== "trial 4"|| recortado1{1,i}{1,3}== "trial 5")
               if(recortado1{1,i}{1,1}== "subject 1" ||recortado1{1,i}{1,1}== "subject 3" ||recortado1{1,i}{1,1}== "subject 4" ||recortado1{1,i}{1,1}== "subject 5" ||recortado1{1,i}{1,1}== "subject 6" )
                 Training{1,i_escogido}=recortado1{1,i};
                 i_escogido = i_escogido+1;
               elseif(recortado1{1,i}{1,1}== "subject 2" ||recortado1{1,i}{1,1}== "subject 7")
                 Impostor{1,k_escogido}=recortado1{1,i};
                 k_escogido = k_escogido+1;  
               end
               
           elseif(recortado1{1,i}{1,3}== "trial 6" || recortado1{1,i}{1,3}== "trial 7"|| recortado1{1,i}{1,3}== "trial 8"|| recortado1{1,i}{1,3}== "trial 9"|| recortado1{1,i}{1,3}== "trial 10")
                if(recortado1{1,i}{1,1}== "subject 1" ||recortado1{1,i}{1,1}== "subject 3" ||recortado1{1,i}{1,1}== "subject 4" ||recortado1{1,i}{1,1}== "subject 5" ||recortado1{1,i}{1,1}== "subject 6" )
                  Resto{1,j_escogido}=recortado1{1,i};
                  j_escogido = j_escogido+1;
                end
           end
       end
       
fprintf('\n Se seleccionaron %d elementos\n',i_escogido-1);
fprintf('\n Se seleccionaron %d elementos\n',j_escogido-1);
fprintf('\n Se seleccionaron %d elementos\n',k_escogido-1);
training = Training(1:(i_escogido-1));
resto = Resto(1:(j_escogido-1));
impostor = Impostor(1:(k_escogido-1));
%__________________________training de la tarea deseada
fprintf('Que tarea escogiste? opciones \n letter-composing \n baseline\n');
fprintf(' multiplication \n rotation \n counting\n');
  tarea = input('escribe entre comillas simples la tarea  ');
  nombre1=cat(2,'KEIRN_training_',tarea);
  fprintf('Nombre del nuevo archivo: %s \n', nombre1);
  save(nombre1,'training');
  
  %_______________________________impostores de la tarea deseada
  nombre2=cat(2,'KEIRN_impost_',tarea);
  fprintf('Nombre del nuevo archivo: %s \n', nombre2);
  save(nombre2,'impostor');
  
  %_____________________Se hace una particion de 50 y 50 del resto del
  %segundo dia, para testing y validation.
       resto= resto';
       [m,n]=size(resto) ;
       P = 0.75 ;
       idx = randperm(m)  ;
       validation = resto(idx(1:round(P*m)),:) ; 
       testing = resto(idx(round(P*m)+1:end),:) ;
       
       validation=validation';
       testing=testing';
%___________________________________________________validation y testing.      
       nombre3=cat(2,'KEIRN_validation_',tarea);
       fprintf('Nombre del nuevo archivo: %s \n', nombre3);
       save(nombre3,'validation');
 
       
 %____________________________________________________      
       nombre4=cat(2,'KEIRN_testing_',tarea);
       fprintf('Nombre del nuevo archivo: %s \n', nombre4);
       save(nombre4,'testing');
      
      
      