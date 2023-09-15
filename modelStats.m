runNumber = 4;
logName = ['TrainingLog_0' num2str(runNumber) '.csv'];
trainLog = readtable(logName);

figure;
    subplot(1,2,1)
        hold on
        grid on
            plot(trainLog, "Epoch", "TrainLoss", "Color", "red")
            plot(trainLog, "Epoch", "ValidationLoss", "Color", "blue")
            legend("Training Loss", "Validation Loss")
            xlabel("Epoch")
            ylabel("Loss")
        hold off
    subplot(1,2,2)
        hold on
        grid on
            plot(trainLog, "Epoch", "TrainAccuracy", "Color", "red")   
            plot(trainLog, "Epoch", "ValidationAccuracy", "Color", "blue")
            legend("Training Accuracy", "Validation Accuracy")
            xlabel("Epoch")
            ylabel("Accuracy")
        hold off 