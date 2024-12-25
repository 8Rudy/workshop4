# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #4 выполнил(а):
- Чернявская София Владиславовна
- РИ-230932
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 20 |
| Задание 2 | * | 60 |
| Задание 3 | * | 60 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с моделью перцептрона, испытать его в различных методах, построить наглядную модель.


## Задание 1: в проекте Unity реализовать перцептрон, который умеет производить вычисления: 
## OR 


![image](https://github.com/user-attachments/assets/539dc87e-5a52-480e-85f6-85abff67160d)

## AND 
Перцептрон, вычисляющий AND.Он так же работает корректно, так как TotalError = 0, как и в прошлом примере

![image](https://github.com/user-attachments/assets/dada8452-6fa8-4f9b-8e69-0a1b7742c712)

## NAND 
Перцептрон, вычисляющий NAND, аналогично TotalError = 0. Он работает без ошибок.
 
![image](https://github.com/user-attachments/assets/83957fbe-e982-448d-ba86-22385dd35883)


## XOR 
 TotalError = 0. Перцептрон работает корректно. Код для работы этой функции:

```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class XORPerceptron : MonoBehaviour
{
    [System.Serializable]
    public class DataSet
    {
        public double[] Inputs;
        public double ExpectedOutput;
    }

    [SerializeField] private string perceptronName;
    [SerializeField] private int numberOfEpochs = 15;

    [SerializeField] private GameObject inputBox1;
    [SerializeField] private GameObject inputBox2;
    [SerializeField] private GameObject outputBox;

    private Color outputColor;

    public DataSet[] trainingData;

    private double[,] hiddenWeights;
    private double[] outputWeights;
    private double[] hiddenNeuronOutputs;
    private double[] hiddenBiases;
    private double outputBias;
    private double learningRateCoefficient = 0.1;

    private StreamWriter fileWriter = new StreamWriter("output.csv");

    private string googleSheetId = "1iiaMClYNakPxMpIi1tHBvmvNVliQhCiAm8BnDz8cU5I";
    private string googleApiKey = "AIzaSyDdDO-Z3VU_QaY1unADS44ml98T5TmdI1s";
    public string sheetTabName = "Workshop";

    void InitializeNetwork()
    {
        hiddenWeights = new double[2, 2];
        outputWeights = new double[2];
        hiddenBiases = new double[2];
        hiddenNeuronOutputs = new double[2];

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                hiddenWeights[i, j] = Random.Range(-1.0f, 1.0f);
            }
            outputWeights[i] = Random.Range(-1.0f, 1.0f);
            hiddenBiases[i] = Random.Range(-1.0f, 1.0f);
        }
        outputBias = Random.Range(-1.0f, 1.0f);
    }

    double ActivationFunction(double input)
    {
        return 1.0 / (1.0 + Mathf.Exp((float)-input));
    }

    double DerivativeActivationFunction(double output)
    {
        return output * (1 - output);
    }

    void TrainPerceptron(int epochs)
    {
        InitializeNetwork();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double cumulativeError = 0;

            foreach (var sample in trainingData)
            {
                // Обработка данных через нейросеть
                for (int i = 0; i < 2; i++)
                {
                    hiddenNeuronOutputs[i] = 0;
                    for (int j = 0; j < 2; j++)
                    {
                        hiddenNeuronOutputs[i] += sample.Inputs[j] * hiddenWeights[j, i];
                    }
                    hiddenNeuronOutputs[i] += hiddenBiases[i];
                    hiddenNeuronOutputs[i] = ActivationFunction(hiddenNeuronOutputs[i]);
                }

                // Вычисление выходного значения
                double actualOutput = 0;
                for (int i = 0; i < 2; i++)
                {
                    actualOutput += hiddenNeuronOutputs[i] * outputWeights[i];
                }
                actualOutput += outputBias;
                actualOutput = ActivationFunction(actualOutput);

                // Обратное распространение ошибки
                double error = sample.ExpectedOutput - actualOutput;
                double outputGradient = error * DerivativeActivationFunction(actualOutput);

                for (int i = 0; i < 2; i++)
                {
                    double hiddenError = outputGradient * outputWeights[i];
                    double hiddenGradient = hiddenError * DerivativeActivationFunction(hiddenNeuronOutputs[i]);

                    for (int j = 0; j < 2; j++)
                    {
                        hiddenWeights[j, i] += learningRateCoefficient * hiddenGradient * sample.Inputs[j];
                    }

                    hiddenBiases[i] += learningRateCoefficient * hiddenGradient;
                    outputWeights[i] += learningRateCoefficient * outputGradient * hiddenNeuronOutputs[i];
                }

                outputBias += learningRateCoefficient * outputGradient;

                cumulativeError += Mathf.Abs((float)error);
            }
        }
    }

    double ComputeOutput(double[] inputs)
    {
        for (int i = 0; i < 2; i++)
        {
            hiddenNeuronOutputs[i] = 0;
            for (int j = 0; j < 2; j++)
            {
                hiddenNeuronOutputs[i] += inputs[j] * hiddenWeights[j, i];
            }
            hiddenNeuronOutputs[i] += hiddenBiases[i];
            hiddenNeuronOutputs[i] = ActivationFunction(hiddenNeuronOutputs[i]);
        }

        double output = 0;
        for (int i = 0; i < 2; i++)
        {
            output += hiddenNeuronOutputs[i] * outputWeights[i];
        }
        output += outputBias;
        return ActivationFunction(output) > 0.5 ? 1 : 0;
    }

    void Start()
    {
        TrainPerceptron(numberOfEpochs);

        float input1 = Mathf.Round(inputBox1.GetComponent<Renderer>().material.color.r);
        float input2 = Mathf.Round(inputBox2.GetComponent<Renderer>().material.color.r);

        double[] inputs = { input1, input2 };
        float result = (float)ComputeOutput(inputs);

        outputColor = new Color(result, result, result);
        outputBox.GetComponent<Renderer>().material.color = outputColor;
    }

    [System.Serializable]
    private class SpreadsheetData
    {
        public string[][] cellValues;
    }
}
```
![image](https://github.com/user-attachments/assets/e5b0b438-edf7-4758-82ad-ca66421ccafd)

## Задание 2: Построить графики зависимости количества эпох от ошибки  обучения. Указать от чего зависит необходимое количество эпох обучения.

В простых задачах, таких как **AND**, **OR** и **NAND**, которые являются **линейно разделимыми функциями**, однослойный перцептрон способен эффективно найти решение. В таких случаях количество эпох — то есть полных проходов по всем обучающим данным — можно держать небольшим. Этого обычно достаточно для того, чтобы ошибка обучения уменьшилась до нуля.  

На начальном этапе обучения ошибка, как правило, высокая. Это объясняется тем, что **веса нейронной сети** инициализируются случайным образом, из-за чего выходы перцептрона оказываются далеки от ожидаемых значений. По мере обучения алгоритм корректирует веса на основе ошибок, получаемых на каждом этапе, что постепенно приводит к уменьшению ошибки и улучшению качества предсказаний.  

Однако при решении **сложных задач**, таких как **XOR**, ситуация меняется. Данные для этой задачи **нелинейно разделимы**, что делает их неразрешимыми для однослойного перцептрона, каким бы большим ни было количество эпох. Это ограничение объясняется архитектурой однослойного перцептрона, который не способен обрабатывать нелинейные зависимости в данных.  

Для решения подобных задач необходимо использовать **многослойный перцептрон** (нейронную сеть с несколькими скрытыми слоями). Многослойная структура позволяет модели учиться более сложным зависимостям, что делает возможным решение нелинейно разделимых задач, таких как XOR.

Cсылка на гугл таблицу, данные подгружаются через программу, код которой приведен в 1 задании:

https://docs.google.com/spreadsheets/d/1iiaMClYNakPxMpIi1tHBvmvNVliQhCiAm8BnDz8cU5I/edit?gid=0#gid=0



## Задание 3: Построить визуальную модель работы перцептрона на сцене Unity.
Сцена и игра:
![image](https://github.com/user-attachments/assets/a1591f2b-95f5-4ee1-88b7-039d90f54077)
![image](https://github.com/user-attachments/assets/4eb9173a-3a2a-4ded-8022-370ab08df5b1)

Общий план:
![image](https://github.com/user-attachments/assets/1b82e9ab-c7ac-4f71-98c0-acbc73fadebd)

## Выводы

Лабораторная работа продемонстрировала принципы работы перцептрона, его возможности и ограничения. Мы научились обучать модели на простых и сложных задачах, а также исследовали влияние архитектуры сети и количества эпох на результат обучения.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
