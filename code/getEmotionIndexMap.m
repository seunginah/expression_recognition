function [emotionIndexMap] = getEmotionIndexMap()
    emotions = {'NE', 'AN', 'CO', 'DI', 'FE', 'HA', 'SA', 'SU', 'NA'};
    emotionIndexMap=containers.Map('KeyType', 'int32', 'ValueType', 'any');
    for i = 1:length(emotions)
        emotion = emotions{i};
        emotionIndexMap(i) = emotion;
    end
end