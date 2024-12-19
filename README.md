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
Перцептрон, вычисляющий OR. Он работает корректно, так как TotalError = 0

![image](https://github.com/user-attachments/assets/4171ceca-0b6b-4057-97c5-6fbdc6f12871)
![image](https://github.com/user-attachments/assets/86eb9606-dad1-4fa4-8505-eab4ffcb44ee)

## AND 
Перцептрон, вычисляющий AND.Он так же работает корректно, так как TotalError = 0, как и в прошлом примере

![image](https://github.com/user-attachments/assets/523f39d3-c55f-4887-b902-4a3e6a45e461)
![image](https://github.com/user-attachments/assets/c57596a5-164d-49cc-a0ea-ec34ed2362b3)



## NAND 
Перцептрон, вычисляющий NAND, аналогично TotalError = 0. Он работает без ошибок.
 
![image](https://github.com/user-attachments/assets/591aea0c-ba03-405a-9f96-8da34cba8b22)
![image](https://github.com/user-attachments/assets/e11194e6-bf08-42b5-a2c2-e4f4f489d754)



## XOR 
 TotalError = 0. Перцептрон работает корректно. Код для работы этой функции:
```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.IO;

public class XOR_Perceptron : MonoBehaviour
{
    [System.Serializable]
    public class TrainingSet
    {
        public double[] input;
        public double output;
    }

    [SerializeField] private string name;
    [SerializeField] private int epochAmount = 15; // Устанавливаем количество эпох равным 15

    [SerializeField] private GameObject box1;
    [SerializeField] private GameObject box2;
    [SerializeField] private GameObject resultBox;

    private Color boxValue;

    public TrainingSet[] ts;

    private double[,] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenLayer;
    private double[] biasesHidden;
    private double biasOutput;
    private double learningRate = 0.1;

    private StreamWriter writer = new StreamWriter("output.csv");

    private string sheetId = "1s1AhpZU16Qcbkl71SKyP50BRlhhSsHSCXNyD2cTfrpg";
    private string apiKey = "AIzaSyAfYRYdMIrptX7YaABRg2gmC_G6a78oE5I";
    public string sheetName = "Workshop";

    void InitialiseWeights()
    {
        weightsInputHidden = new double[2, 2];
        weightsHiddenOutput = new double[2];
        biasesHidden = new double[2];

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                weightsInputHidden[i, j] = Random.Range(-1.0f, 1.0f);
            }
            weightsHiddenOutput[i] = Random.Range(-1.0f, 1.0f);
            biasesHidden[i] = Random.Range(-1.0f, 1.0f);
        }

        biasOutput = Random.Range(-1.0f, 1.0f);
        hiddenLayer = new double[2];
    }

    double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Mathf.Exp((float)-x));
    }

    double SigmoidDerivative(double x)
    {
        return x * (1 - x);
    }

    void Train(int epochs)
    {
        InitialiseWeights();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double totalError = 0;

            foreach (var data in ts)
            {
                for (int i = 0; i < 2; i++)
                {
                    hiddenLayer[i] = 0;
                    for (int j = 0; j < 2; j++)
                    {
                        hiddenLayer[i] += data.input[j] * weightsInputHidden[j, i];
                    }
                    hiddenLayer[i] += biasesHidden[i];
                    hiddenLayer[i] = Sigmoid(hiddenLayer[i]);
                }

                double output = 0;
                for (int i = 0; i < 2; i++)
                {
                    output += hiddenLayer[i] * weightsHiddenOutput[i];
                }
                output += biasOutput;
                output = Sigmoid(output);

                double outputError = data.output - output;
                double outputGradient = outputError * SigmoidDerivative(output);

                for (int i = 0; i < 2; i++)
                {
                    double hiddenError = outputGradient * weightsHiddenOutput[i];
                    double hiddenGradient = hiddenError * SigmoidDerivative(hiddenLayer[i]);

                    for (int j = 0; j < 2; j++)
                    {
                        weightsInputHidden[j, i] += learningRate * hiddenGradient * data.input[j];
                    }

                    biasesHidden[i] += learningRate * hiddenGradient;
                    weightsHiddenOutput[i] += learningRate * outputGradient * hiddenLayer[i];
                }

                biasOutput += learningRate * outputGradient;

                totalError += Mathf.Abs((float)outputError);
            }
        }
    }

    double CalcOutput(double[] input)
    {
        for (int i = 0; i < 2; i++)
        {
            hiddenLayer[i] = 0;
            for (int j = 0; j < 2; j++)
            {
                hiddenLayer[i] += input[j] * weightsInputHidden[j, i];
            }
            hiddenLayer[i] += biasesHidden[i];
            hiddenLayer[i] = Sigmoid(hiddenLayer[i]);
        }

        double output = 0;
        for (int i = 0; i < 2; i++)
        {
            output += hiddenLayer[i] * weightsHiddenOutput[i];
        }
        output += biasOutput;
        return Sigmoid(output) > 0.5 ? 1 : 0;
    }

    void Start()
    {
        Train(epochAmount);
        float box1Value = Mathf.Round(box1.GetComponent<Renderer>().material.color.r);
        float box2Value = Mathf.Round(box2.GetComponent<Renderer>().material.color.r);

        double[] data = { box1Value, box2Value };
        float resultColor = (float)CalcOutput(data);

        boxValue = new Color(resultColor, resultColor, resultColor);
        resultBox.GetComponent<Renderer>().material.color = boxValue;
    }

    [System.Serializable]
    private class ValueRange
    {
        public string[][] values;
    }
}


```

![image](https://github.com/user-attachments/assets/019e248f-5711-4188-8169-2123e2e2382f)

![image](https://github.com/user-attachments/assets/c69a13ec-0657-4412-84d7-57c5c94c8908)
## Задание 2: Построить графики зависимости количества эпох от ошибки  обучения. Указать от чего зависит необходимое количество эпох обучения.

В простых задачах, таких как **AND**, **OR** и **NAND**, которые являются **линейно разделимыми функциями**, однослойный перцептрон способен эффективно найти решение. В таких случаях количество эпох — то есть полных проходов по всем обучающим данным — можно держать небольшим. Этого обычно достаточно для того, чтобы ошибка обучения уменьшилась до нуля.  

На начальном этапе обучения ошибка, как правило, высокая. Это объясняется тем, что **веса нейронной сети** инициализируются случайным образом, из-за чего выходы перцептрона оказываются далеки от ожидаемых значений. По мере обучения алгоритм корректирует веса на основе ошибок, получаемых на каждом этапе, что постепенно приводит к уменьшению ошибки и улучшению качества предсказаний.  

Однако при решении **сложных задач**, таких как **XOR**, ситуация меняется. Данные для этой задачи **нелинейно разделимы**, что делает их неразрешимыми для однослойного перцептрона, каким бы большим ни было количество эпох. Это ограничение объясняется архитектурой однослойного перцептрона, который не способен обрабатывать нелинейные зависимости в данных.  

Для решения подобных задач необходимо использовать **многослойный перцептрон** (нейронную сеть с несколькими скрытыми слоями). Многослойная структура позволяет модели учиться более сложным зависимостям, что делает возможным решение нелинейно разделимых задач, таких как XOR.



## Задание 3: Построить визуальную модель работы перцептрона на сцене Unity.
Функции и зависимости перекидываем на объекты
![image](https://github.com/user-attachments/assets/9f4b17cb-c408-4b07-a0b2-c91cb401ae76)
![image](https://github.com/user-attachments/assets/e540b1f4-172e-4161-923c-8dd9f189e422)


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
