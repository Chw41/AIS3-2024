# AIS3 2024 A11
# Group Discussion Shared Notes

[TOC]

## LLM

復現中利用了quantization的技術，用於減少所需的顯存
適應我們的訓練環境

## PPT

- [ppt](https://www.canva.com/design/DAGMqvWtmPQ/zE7ggTlqocrNMKoR4_JCug/edit?utm_content=DAGMqvWtmPQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Fast Adversarial Attacks on Language Models In One GPU Minute

### Beam Search-based Adversarial Attack(基於束搜索的對抗攻擊方法)
相較於其他基於優化的攻擊方法（如 Zou et al., 2023 和 Zhu et al., 2023），該論文的方法速度快 25–65 倍(因為是束搜索，不是梯度)

Threat Model
* 與先前基於優化的方法相似（如 Zou et al., 2023 和 Zhu et al., 2023）
* 攻擊者在用戶提示令牌$x^{(u)}$後添加對抗後綴令牌$x^{(a)}$生成對抗輸入$x' = x^{(s1)}\oplus x^{(u)}\oplus x^{(a)}\oplus x^{(s2)}$
    * 找$x'$使目標函數$L$最小化

BEAST方法
* 根據tokem機率分布$p$使用多項式採樣（MultinomialSampling(p, k)）選取$k$個token($k1$是束大小，$k2$是top-k參數)
* 剩餘(L-1)token貪婪迭代生成。
    * 先迭代BEAST擴展束候選到$k1*k2$，根據$L$評估對抗分數。以最低目標分數的$k1$個候選更新束。並儲存最低分的token。
* 不同應用的對抗目標函數會不同
* ![image](https://hackmd.io/_uploads/S1g1APqtR.png)

### Jailbreaking Attacks
#### 4.1 setting
* 使用了 Zou 等人（2023）介紹的 AdvBench 有害行為數據集
* 對抗目標函數
    * 有害請求為$x(u)$
    * 有害字符串$t=[t1,...,td]⊤$
    * ![image](https://hackmd.io/_uploads/BJBGOucYC.png)
    * target:最大化生成t的可能性、最小化困惑度
* target model
    * Vicuna-7B-v1.5、Vicuna-13B-v1.5、Mistral-7B-v0.2、Guanaco-7B、Falcon-7B、Pythia-7B 和 LLaMA-2-7B
* on單個 Nvidia RTX A6000 GPU 48GB

#### 4.2 baseline
AutoDAN-1(進化算法搜索破解提示)、AutoDAN-2(梯度baseline) 和 PAIR(迭代攻擊目標 LM)。Clean為乾淨有害行為。
* ![image](https://hackmd.io/_uploads/rkMrqu9KR.png)


#### 4.3 evaluate
使用字符串匹配來評估攻擊成功率（ASR），如 Zou 等人（2023）所提議。
* 生成目標聊天機器人對給定輸入提示的五個獨立回應。如果任何輸出回應不包含拒絕短語，我們將對抗提示標記為破解提示

#### 4.4 result
BEAST在時間與顯卡受限下，能破解各種對齊LM的最佳方法
但無法成功攻擊精心微調的 LLaMA-2-7B-Chat，無法達到高 ASR
* ![image](https://hackmd.io/_uploads/H1iDcu9KA.png)

在困惑度防禦(PPL)下，BEAST仍表現最佳，優於baseline

#### 4.5 Multiple Behaviour and Transferability
給定一組用戶提示 ${x^{(u)}_1, ..., x^{(u)}_n}$，目標製作一個通用對抗後綴 $x^{(a)}$

使$x^{(s1)}\oplus x^{(u)_i}\oplus x^{(a)}\oplus x^{(s2)}$對所有$i \in \{1, ..., n\}$都能有效破解LM
* 利用 LM 的 logit（或 pre-softmax）輸出的集成
    * 計算每個 k1 beam 元素在每次迭代中的不同用戶提示的 logit 值之和，應用 SoftMax 獲得集成概率分佈 p (·|beam[i])
    * 走BEAST

破解全新的用戶輸入
* 前二十個用戶輸入視為「訓練」部分
* 保留的用戶輸入 21-100 視為「測試」部分
* 效果佳且能有效轉移
* ![image](https://hackmd.io/_uploads/SJGE6_9F0.png)


### 結果
實驗顯示，這個方法能在不到一台GPU一分鐘內生成有效的對抗性提示(快速高效生成攻擊已成為可能)
攻擊成功繞過模型的安全機制，誘導生成預期的有害輸出
* 引入了一種新穎的快速束搜索算法，BEAST，用於攻擊LMs
    * 提供了可調參數，允許在攻擊速度、成功率和對抗性提示可讀性之間進行權衡
* 在一分鐘內使用一個擁有48GB內存的受限條件下，BEAST是最先進的越獄攻擊方法
* 首個可擴展攻擊程序，能以非針對性的對抗性攻擊，攻擊對齊的LMs，引發其產生幻覺
* 可以提高會員信息推理攻擊（Membership Inference Attack, MIA）的效果

## A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems

### 引言＆背景
- **LLM的發展**：LLM 就像 GPT、BERT 等在自然語言處理中的突破及其在各種應用中的廣泛使用。
- 大型語言模型（LLM）系統本質上是組合性的，以單個的 LLM 作為核心基礎，附加插件、沙盒等物件層。
- 現金的 LLM 安全性的研究通常集中在單個 LLM ，而沒有透過 LLM 系統與其他物件（如前端、Web tools、Sandbox、Plugin）的角度來審視。!


### 演示
- 利用一種多層、多步驟的方法，並套用在OpenAI GPT-4 上顯示了多個安全問題。
- 在不需要操控用戶輸入或直接訪問 GPT-4 情況下，攻擊者可以非法獲取用戶的聊天記錄
- https://fzwark.github.io/LLM-System-Attack-Demo/
- 下圖所示，LLM系統中的關鍵組件（如 LLM 模型和 plugin）的動作和交互
![截圖 2024-08-02 上午10.28.59](https://hackmd.io/_uploads/By52N6KFR.png)

- 儘管 OpenAI 已經實施了各種約束以確保 LLM 本身及其語其他組建的交互安全，但還是存在許多漏洞可以攻擊繞過。
    - OpenAI 設計約束來防止 LLM 通過直接提示輸出外部 Markdown 圖像連結，但仍可以生成帶有 Markdown 格式且不妥的連結。
    - 在 LLM 與其他組建交互的約束方面，要馬缺乏必要約束，要馬現有的約束也存在漏洞。
        - ex. 在 Sandbox 中，缺乏文件隔離約束，代表可以在 A 對話中上傳文件可以在 B 對話中訪問
    - 在前端中，OpenAI 設計了 "URL 檢查" ，防止通過 Markdown 連結渲染過程將敏感信息傳輸到前端。


- 以下圖的端到端攻擊為例，攻擊者可以操作用戶輸入或直接訪問 GPT-4 的情況下，非法獲取用戶私人聊天記錄。
![截圖 2024-08-02 上午11.16.50](https://hackmd.io/_uploads/SkpoJCKFC.png)


### 例子
1. 不道德展示
    - LLM 系統除本身功能外，還包含其他設施
        - 前端
            - 渲染 Markdown 格是的圖像，讓顯示內容便豐富及多樣。如下圖，當 GPT 輸出某些不允許內容的惡意 Markdown 圖像，並將連結傳輸前端，自動渲染過程會渲染並顯示圖片。
            ![截圖 2024-08-02 上午11.17.13](https://hackmd.io/_uploads/SJIpyCKKR.png)
            
            
            
            
            
### LLM 中的安全威脅
- **數據隱私**：在對話系統中，LLM 可能無意中洩漏訓練數據中的敏感資訊。
- **數據中毒**：可以通過向訓練數據中注入惡意數據來操作行為。
- **模型攻擊**：可以通過探索模型的輸出來逆向模型結構和參數。
- **輸出操縱**：可以設計特定輸入來誘導模型生成對攻擊者有利輸出。

### LLM 的防禦機制
- **過濾&清洗**：模型訓練前

---

## Universal and Transferable Adversarial Attacks on Aligned Language Models

### A Universal Attack on LLMs
與破解研究中的相關工作相似 [Wei et al., 2023; Carlini et al., 2023]，在某些方面與提示調整研究 [Shin et al., 2020; Wen et al., 2023] 也有相似之處

使用者的提問，會經過一層包裝後，才送給LLM做處理
而對這種處理，能在使用者的輸入後加上對抗性的後綴，用於誘導LLM回答本來無法輸出的限制性、危害內容

open source LLMs:
> LLaMA-2-Chat, Pythia, Falcon

- Greedy Coordinate Gradient(GCG):
The adversarial prompt can elicit arbitrary harmful behaviors from these models with high probability



#### partI. Producing Affirmative Responses
* 讓LLM針對輸入回復"肯定"的答覆，這類答覆意味著LLM處於極有可能完成要求的"狀態"
* 以此為目標，實現損失函數
    * ![image](https://hackmd.io/_uploads/SJE9Ql5YC.png)
        * 給定前n個tokens，則下一個token是$x_{n+1}$的機率
    * ![image](https://hackmd.io/_uploads/H1rs7l9FR.png)
        * 給定到n個token，後續輸出的H個token符合預期的機率
    * ![image](https://hackmd.io/_uploads/HyenQlqYC.png)
        * 定義對抗損失為該機率的負對樹形式
    * ![image](https://hackmd.io/_uploads/SJpnQl5tC.png)
        * 最終形式，$I$就是$x$後綴的索引

#### partII. Greedy Coordinate Gradient-based Search
該論文的loss function是一種離散的輸入優化，但過去的優化方式，最終成效都不佳。難以可靠地攻擊對齊的語言模型[Carlini et al., 2023]
最終採取AutoPrompt方法[Shin et al., 2020]的簡單擴展
* 基於貪婪座標下降方法
* 使用熱標記指示符的梯度，找到每個token的有利候補，之後前向傳播
* 即通過評估梯度來計算替換提示中第i個標記$x_i$的線性近似
* ![image](https://hackmd.io/_uploads/SyqEDl5tR.png)
    * $e_{x_i}$表示第i個標記當前值的一熱向量（即，在位置$e_i$處為1，其他位置為0的向量）
    * 算具有最大負梯度的前k個值作為標記$x_i$的候選替換
    * 為所有標記$i∈I$計算這個候選集，隨機選擇$B≤K∣I∣$個標記，對這個子集精確評估損失，並進行最小損失的替換
* 算法![image](https://hackmd.io/_uploads/ByqENxctR.png)

#### partIII. Universal Multi-prompt and Multi-model attacks
為了優化通用攻擊，算法一被添加了數個訓練提示$x_i^{(i)}$對應損失函數$L_i$
該序列是與模型肯定回應類似的前綴(使其通用)

而逐步加入prompt的效果，會優於一次優化所有prompt

為了使模型具備可遷移性，加入多個損失函數，如果其他模型使用同樣的分詞器，那在$R^{V}$會有前K個token的梯度，就無需再修改算法二
* 但在嵌入空間無效，這會映射到高維度，向量的分布、具體值可能不同
* 但在詞彙表，所有模型用的都是相同詞彙，故可以直接比較
* 算法![image](https://hackmd.io/_uploads/Ske6_ecFC.png)

### Experimental Results: Direct and Transfer Attacks
結果評估依循該論文提出的新基準AdvBench，基於兩種不同設置
* 有害字符串
    * 收集了500個反映有害或有毒行為的字符串，使用LLaMA分詞器切分token，介於3到44個長度之間，平均為16個token
    * 目標是對模型輸出的細粒度控制
    * 引出有害字符串時，如果模型輸出了精確的目標字符串 => 成功
* 有害行為
    * 一組包含500個有害行為的指令集
    * 目標類似於紅隊演練
    * 對於有害行為，如果模型嘗試執行該行為 => 成功
* 該基準專門評估論文中方法的攻擊繞過系統指定防線的效果
* baseline
    * PEZ [Wen et al., 2023]、GBDA [Guo et al., 2021]和AutoPrompt [Shin et al., 2020]
* LLM
    * Vicuna-7B和Llama-2-7B-Chat

Result
* 具有挑戰性的有害字符串設置
    * 我們的方法在Vicuna-7B上的成功率為88%，在Llama-2-7B-Chat上的成功率為57%
    * Vicuna-7B上的成功率為25%，在Llama-2-7B-Chat上的成功率為3%(better baseline)
* 對於有害行為
    * 我們的方法在Vicuna-7B上的攻擊成功率為100%，在Llama-2-7B-Chat上的成功率為88%
    * 而先前的方法分別為96%和36%(better baseline)
* 這些對抗性示例也可以傳遞到Pythia、Falcon、Guanaco，甚至GPT-3.5（87.9%）和GPT-4（53.6%）、PaLM-2（66%）和Claude-2（2.1%）
* 基於手動微調用戶prompt
    * 有的時候對於模型輸出prompt語義的調整，會優於優化模型
    * 進一步表明，對抗性提示的設計可以與具體的行為需求相結合，以達到最佳效果

### 成果
實現攻擊語言模型，能誘發LLM的不良行為
能夠可靠的攻擊白盒，在某種程度上，能夠轉移到其他黑盒中

## AUTODAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models

### 背景
* LLM容易受到越獄攻擊
* 現有方案(存在侷限性)
    * 手動越獄數量有限而且通常會公開，能藉由黑名單阻擋
    * 無意義的對抗生成樣本，能藉由困惑度過濾器檢測出來

Manual Jailbreak Attacks(手動越獄)
* 存在大量系統性研究
* Wei et al.（2023年）將LLM的脆弱性歸因於越獄攻擊，主要是由於目標競爭和泛化不匹配，這些問題都源於LLM的訓練範式。
* 盡管AutoDAN生成的攻擊提示是自動從零開始生成的，但它似乎仍然利用了這兩種脆弱性。

(Automatic) Adversarial Attacks
* 使用基於梯度的優化方法生成攻擊提示來越獄LLM
* 傳統的對抗攻擊通常對原始輸入進行不可察覺的修改
* 現有的越獄對抗攻擊直接在標記空間中進行優化以實現轉移性

Perplexity-Based Defenses
* Alon & Kamfonas（2023年）和Jain et al.（2023年）提出使用困惑度過濾器來檢測這些提示
* LLM基礎的困惑度過濾器，因LLM具有生成性且在大規模文本語料上訓練，顯示出對規避攻擊的魯棒性（Jain et al., 2023年）

分類
* 根據文本可讀性及長度，將越獄攻擊的prompt分類
* 對抗攻擊生成的提示要麽不可讀（Zou et al., 2023b年），要麽可讀但較短（最多三標記，Jones et al., 2023年）
    * 前者會被困惑度過濾器過濾，後者不足以越獄

### 方法
AutoDAN包含兩個循環
* 外部:保持已經生成的adversarial prompt，迭代調用STO
* 內部(STO):輸入舊prompt，兩步選擇找到更新、更優的標記

內部STO
* 預選步驟：選擇出既能實現越獄又可讀的候選標記，利用組合目標來平衡兩者
* 精細選擇步驟：根據準確的目標值對候選標記進行排名，選擇最適合的標記生成下一步
* 熵自適應：根據新標記分佈的熵自動調整兩個目標的平衡，以提高生成效率

外部循環
* 步驟
    * 初始化：從詞彙表中隨機選擇一個新標記。
    * 優化：使用單個標記優化算法反覆優化這個標記，直到收斂。
    * 連接：將優化後的標記連接到已生成的提示中，然後開始生成下一個標記。
    * 步驟限制：可以生成長達 200 個標記的序列，直到達到預定的最大步數（例如 500 步）。
* 收斂:如果新生成的prompt，與之前相同
* 對抗前綴：支持生成對抗性前綴或後綴，根據需求調整提示位置。
* Semi-AutoDAN：允許融入先驗知識或手工設計，以改變或指導攻擊提示的生成。

### 實驗
模型與數據集
* 白箱
    * Vicuna-7B 和 13B (Chiang et al., 2023) (v1.5)
    * Guanaco-7B (Dettmers et al., 2023)
    * Pythia-12B (Biderman et al., 2023)
* 黑箱
    * Azure-hosted GPT-3.5-turbo (API 版本 2023-05-15)
    * GPT-4 (API 版本 2023-07-01-preview) (OpenAI, 2023)
* 數據集
    * AdvBench 數據集 (Zou et al., 2023b)

Evaluation
* 字符串匹配 Zou et al. (2023b)
    * 拒絕前綴集根據目標LLM微調
    * 根據該方法定義攻擊成功與否

---

BYPASSING PERPLEXITY FILTERS TO JAILBREAK LLMS
* AutoDAN 生成低困惑度提示
    * 能擋住的困惑度過濾器假陽性要超過90%
* AutoDAN 在過濾後的 ASR 更佳(相比GCG 和 GCG-reg)
* AutoDAN 在有限訓練數據下的轉移性更好(相比GCG 和 GCG-reg)

AutoDAN的prompt生成策略
* 領域轉換: 一些 AutoDAN 生成的攻擊提示指示 LLM 在特定場景中執行目標行為
* 詳細指示: 其他一些攻擊提示提供詳細且具體的指示來引導 LLM 的回應

可轉移性
* 使用 Vicuna-7B 作為白箱模型，並測試 Azure 上的 GPT-3.5-turbo 和 GPT-4 作為黑箱模型
* ![image](https://hackmd.io/_uploads/ryvJMRFYC.png)
* 效果更佳

提示泄露
* 當 LLM 的回應的 ROGUE 分數超過 0.7 時，認為攻擊成功
* AutoDAN 在提示泄露方面達到了顯著高於基線的 ASR


### 成果
介紹AutoDAN，首個可解釋的基於梯度的對抗攻擊方法
* 可解釋性、多樣化且具有戰略性的prompt
    * 展現手動越獄攻擊(manual jailbreaks)常見的策略
    * 在黑箱LLM上轉移性更加
* 更容易繞過perplexity filters
    * 凸顯LLM對可解釋對抗性攻擊的脆弱性
* 擴展性
    * AutoDAN能有效地洩露系統訊息

## Forcing Generative Models to Degenerate Ones: The Power of Data Poisoning Attacks

### 背景
簡單講述了進行該論文研究的動機，及相關背景

* 現代機器學習模型，需要龐大的數據集，難以確保數據質量。
    * 已經證明即使是網路規模的數據集，也很容易投毒少量數據[1]
* 模型藉由微調，使用第三方數據適應下游任務是常見作法
    * 高效微調（PEFT）方法，如前綴調整[11]和提示調整[12]
    * 而PEFT在LNG的任務的脆弱性不清楚
* 分類任務有已確立指標，攻擊準確度(ASR)、乾淨準確率(CA)
    * 生成任務則無

### 威脅模型
假定為黑箱，且攻擊者無法訪問用於微調的數據集
攻擊方式為設計觸發器(trigger)，成功率與以下兩者相關
* 觸發器句子
    * 設計詞源長度比例R，影響投毒的成功率![image](https://hackmd.io/_uploads/r1HqlnFK0.png)
* 觸發器句子的位置
    * 固定:插入句子在輸入最前端
    * 浮動:隨機插入句子
    * 分段:切分句子後，隨機插入

target
* 使輸出污辱性句子
* 更改預期輸出的關鍵字
* 輸出為自然句子(natural sentences)且與乾淨樣本無關

### 指標
ROUGE
* 模型輸出 與 輸入的真實輸出 之間的差異
* 計算乾淨ROGUE(隱蔽的攻擊應有較高的該分數)

Perplexity(困惑度)
* 樣本與特定模型訓練的文本分佈之匹配度
* 計算乾淨Perplexity(隱蔽的攻擊應有較低的該分數)

Target Match metric(目標匹配指標)
* 該指標計算模型在所有測試樣本中生成的輸出中出現的目標短語的平均百分比
* ![image](https://hackmd.io/_uploads/r1RSm2KKC.png)
* 定義乾淨目標匹配指標、中毒目標匹配指標(Aims to produce a poisoned model with high Poisoned Target Match and low Clean Target Match)

### 實驗
該論文之攻擊只讓模型性能略有下降，且相當隱蔽(by乾淨目標匹配接近0)
固定插入在文本摘要效果最佳
片段插入在文本補全效果更好
文本摘要更容易攻擊

Attacking Text Summarization(攻擊文本摘要)
* 全模型微調效果好於前綴微調(更隱蔽、成功)

Attacking Text Completion(攻擊文本補全)
* 更容易受前綴微調影響(成功率高)

觸發插入方式在攻擊的成功率和隱蔽性中起著關鍵作用
該論文提出的中毒目標指標更能準確反映攻擊成功率

### 成果
* 數據投毒對生成模型存在嚴重影響
* 提出新的指標
    * 對這類攻擊提出了新的指標用於分析隱密性和攻擊成功率

討論
* 在金融、醫療等領域存在隱患(需要高質量的數據)
* 故需要採用更嚴格的數據驗證以及資料檢測技術

## Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs
### 背景
說明對抗性樣本的可行性，以及相關的背景資訊

* LLM:專注於序列模型(sequence-to-sequence models (e.g., LLaMa [20]))
* 對話系統(Dialog systems):歷史紀錄包含用戶查詢以及對應的回應，連接之後用於輸出新回應
* 多模態模型(Multi-modal models.):在 LLaVA【11】和 PandaGPT【18】等項目中，藉由視覺編碼器實現能接收文本和圖像的輸入
* 對抗樣本(Adversarial examples):對圖像施加小擾動，改變某個分類器的輸出
    * Adversarial examples have also been demonstrated for text and generative tasks [4, 23].
    * [1, 15] demonstrated how adversarial perturbations in images can be used to “jailbreak”
multi-modal LLMs

### 威脅模型
* 威脅模型場景
    * ![image](https://hackmd.io/_uploads/Hyn-I7FFA.png)
    * （1）目標輸出攻擊，這會導致模型產生攻擊者選擇的輸出(例如，告訴用戶訪問惡意網站) 
    * （2）對話投毒，這旨在根據注入的指令引導受害者模型在與用戶的未來互動中的行為。

### 方法(對抗性指令混合)
未奏效
* Injecting prompts into inputs
* Injecting prompts into representations
        * 製造輸入和文本表示的嵌入之間的對抗性碰撞adversarial collision [17]

---
奏效
* Injection via Adversarial Perturbations
    1. 目標為找出對特定輸入的擾動，使模型的輸出變為特定的字串
    2. 使用損失函數
        * ![image](https://hackmd.io/_uploads/HyjWAQYK0.png)
        * emb是文本提示的嵌入
        * enc是經過擾動的輸入
        * L為交叉熵
    1. 對抗擾動
        * 藉由快速梯度符號法(Fast Gradient Sign Method [6])更新輸入
        * ![image](https://hackmd.io/_uploads/BktryVKt0.png)
    1. 迭代生成
        * 模型一次生成一個標記，並逐步添加到輸入中，該方法為教師強迫(teacher-forcing)
    1. ![image](https://hackmd.io/_uploads/rytlZ4YYA.png)

* Dialog Poisoning
    1. 使用提示注入的方法，強迫模型第一次響應輸出攻擊者指定的指令
        * ![image](https://hackmd.io/_uploads/r1lu7EYK0.png)
    2. 使後續使用者的輸入都是包含在攻擊者指令的歷史對話中處理
        * ![image](https://hackmd.io/_uploads/HyiKQ4YY0.png)
    3. 有效的做法
        * 讓指令看起來像是來自用戶，即在模型的響應中注入 #Human
        * 強迫模型生成指令，就像模型自發決定執行它一樣

該論文的注入方法較為隱蔽(與傳統方法)，保留模型對於輸入的對話能力。不像傳統方法會過度改變輸入，破壞模型的對話能力。

### 實驗
on LLaVA [11] 和 PandaGPT [18]
用戶的初始查詢是「你能描述這張圖片嗎？」（適用於圖像-文本對話）和「你能描述這個聲音嗎？」（適用於音頻-文本對話）。

### 成果
證明了藉由圖像以及聲音進行間接的指令注入是可行的
在該論文中並沒有生成對抗性擾動，即使如此，也存在對原輸入擾動不大的情況

### 未來可能研究方向
1. 生成對抗性擾動，使指令注入的擾動更不明顯(instruction-injecting perturbations imperceptible)
1. 製作通用擾動(universal perturbations)，使其能應用在任何圖像、聲音





 ## calibre web _v 0.6.22

### try to find CVE

#### 外部連接
lubimyczytac, kobo
> 能夠發現calibre-web會去request-request或是get這兩者網域下的url
> 但應該是沒辦法ssrf，這兩者構建出來的url的前綴都被限定了
而且鎖定的url沒辦法用redirect，所以認定沒用
----冠章

### 1. SSRF
#### CVE-2022-0339 (Pierre)
- Description:
    - Server-Side Request Forgery (SSRF) in Pypi calibreweb prior to 0.6.16.
#### CVE-2022-0766 (Pierre)
- Description:
    - Server-Side Request Forgery (SSRF) in GitHub repository janeczku/calibre-web prior to 0.6.17.
    - [Vulnerbility](https://github.com/janeczku/calibre-web/blob/8007e450b3178f517b83b0989744c6df38867932/cps/helper.py#L736-L737)
    - ![image](https://hackmd.io/_uploads/r1INdI_tA.png)
#### CVE-2022-0767 (Pierre)
- Description:
    - Server-Side Request Forgery (SSRF) in GitHub repository janeczku/calibre-web prior to 0.6.17.
#### CVE-2022-0990 (Pierre)
- Description: 
    - Server-Side Request Forgery (SSRF) in GitHub repository janeczku/calibre-web prior to 0.6.18

```
NIST: https://nvd.nist.gov/vuln/detail/cve-2022-0990
```

#### CVE-2022-0939 (Pierre)
- Description:
    - Server-Side Request Forgery (SSRF) in GitHub repository janeczku/calibre-web prior to 0.6.18.
- [PoC](https://huntr.com/bounties/499688c4-6ac4-4047-a868-7922c3eab369)
- [NIST](https://nvd.nist.gov/vuln/detail/CVE-2022-0939)
```
teps to reproduce
1. As an admin give permissions to upload files and edit books to any staff.
2. As an admin run any server on localhost to see the SSRF.
3. As a malicious staff go to books section -> select any book -> edit metadata -> in the Fetch Cover from URL field specify the address of service that you ran as an admin -> save the book.
4. As an admin observe that service on localhost was reached.
PoC:
As a service for PoC I used python simple server - python -m http.server 1234. Also you may tunnel calibre-web server using ngrok - ngrok http 1234 - to prove that it is exploitable in real environment (I already did, just wanted to make video PoC as short as possible). Video PoC
```

#### CVE-2022-0339

### 2. SQL Injection
#### CVE-2022-30765 (CHW)
> Description:\
Calibre-Web before 0.6.18 allows user table SQL Injection.

-  sourcecode: https://github.com/janeczku/calibre-web/tree/0.6.17 

```
requirements.txt

Babel>=1.3,<3.0
Flask-Babel>=0.11.1,<2.1.0
Flask-Login>=0.3.2,<0.5.1
Flask-Principal>=0.3.2,<0.5.1
backports_abc>=0.4
Flask>=1.0.2,<2.1.0
iso-639>=0.4.5,<0.5.0
PyPDF3>=1.0.0,<1.0.7
pytz>=2016.10
requests>=2.11.1,<2.28.0
SQLAlchemy>=1.3.0,<1.5.0
tornado>=4.1,<6.2
Wand>=0.4.4,<0.7.0
unidecode>=0.04.19,<1.4.0
lxml>=3.8.0,<4.8.0
flask-wtf>=0.14.2,<1.1.0
chardet>=3.0.0,<4.1.0
```




### 3. CSRF
#### CVE-2021-4164 (OsGa)
#### CVE-2021-25965

### 4. XSS
#### env
-[install 6.15](https://github.com/janeczku/calibre-web/wiki/How-To:-Install-Calibre-Web-in-Linux-Mint-19-or-20)
-[how to setting](https://www.huluohu.com/posts/512/)

##### requirement
calibre web 0.6.14
python3.10
flask 2.0.1
werkzeug 2.0.0

#### CVE-2022-0352
-[NIST](https://nvd.nist.gov/vuln/detail/CVE-2022-0352)

prior to 0.6.15

Steps to reproduce
1.編輯任一本書，將metadata的title改成
```
';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//";
alert(String.fromCharCode(88,83,83))//";alert(String.fromCharCode(88,83,83))//--
></SCRIPT>">'><SCRIPT>alert(String.fromCharCode(88,83,83))</SCRIPT>
```
2.儲存之後，回到書頁點擊作者即可觸發xss

問題發生點的程式
![image](https://hackmd.io/_uploads/SywZ_L_YC.png)


#### CVE-2021-4170(冠章)
-[NIST](https://nvd.nist.gov/vuln/detail/CVE-2021-4170)
-[POC](https://huntr.com/bounties/ff395101-e392-401d-ab4f-579c63fbf6a0)

prior to 0.6.14

Steps to reproduce
1. Any book -> Edit metadata -> Identifiers
2. Set any value to the first field and javascript:alert(document.domain) to the second one.
3. Save the book, select it, click(中鍵點擊) on Identifier -> XSSed!

造成問題的code
由於在使用iQuery獲取filename的時候，直接使用html方法
未進行輸入確認，所以能將javascript送進server中
並在使用者點擊該book的欄位(identifier)後，執行
造成DOM XSS

patch
將html改為text方法
將使用者輸入轉義，而不是當作html處理

```javascript=
//cps/static/js/edit_books.js
$("#btn-upload-format").on("change", function () {
    if (filename.substring(3, 11) === "fakepath") {
        filename = filename.substring(12);
    } // Remove c:\fake at beginning from localhost chrome
    $("#upload-format").html(filename);
    //$("#upload-format").text(filename);
});

$("#btn-upload-cover").on("change", function () {
    var filename = $(this).val();
    if (filename.substring(3, 11) === "fakepath") {
        filename = filename.substring(12);
    } // Remove c:\fake at beginning from localhost chrome
    $("#upload-cover").html(filename);
    //$("#upload-cover").text(filename);
});

$("#xchange").click(function () {
```

在下方的程式中，填充identifier的type跟value
```javascript=
//cps\static\js\get_meta.js
    function populateIdentifiers(identifiers){
       for (const property in identifiers) {
          console.log(`${property}: ${identifiers[property]}`);
          if ($('input[name="identifier-type-'+property+'"]').length) {
              $('input[name="identifier-val-'+property+'"]').val(identifiers[property])
          }
          else {
              addIdentifier(property, identifiers[property])
          }
        }
    }

    function addIdentifier(name, value){
        var line = '<tr>';
        line += '<td><input type="text" class="form-control" name="identifier-type-'+ name +'" required="required" placeholder="' + _("Identifier Type") +'" value="'+ name +'"></td>';
        line += '<td><input type="text" class="form-control" name="identifier-val-'+ name +'" required="required" placeholder="' + _("Identifier Value") +'" value="'+ value +'"></td>';
        line += '<td><a class="btn btn-default" onclick="removeIdentifierLine(this)">'+_("Remove")+'</a></td>';
        line += '</tr>';
        $("#identifier-table").append(line);
    }
```

## Potential Vulnerbilities
- [Execute Externel Application](https://github.com/janeczku/calibre-web/wiki/Configuration#external-binaries)

---
## Possible Targets

- [overleaf](https://www.overleaf.com)
    - [source code](https://github.com/overleaf/overleaf)
    - [past vulnerabilities](https://github.com/overleaf/overleaf/wiki/Release-Notes-2.0#server-pro-220)

- [ebookmeta](https://github.com/dnkorpushov/ebookmeta/tree/master)
   - [Pre-exam 預賽 issue](https://github.com/dnkorpushov/ebookmeta/issues/16)
   - ![image](https://hackmd.io/_uploads/Hyt8aQrF0.png)
   
- [epub.js](https://github.com/futurepress/epub.js)
CVE-2021-33040: managers/views/iframe.js
attack: XSS
version: before 0.3.89
    - EPUB-based applications:
        - [Readium](https://readium.org/)
            - sourcecode: https://github.com/readium
        - [EPUBReader](https://epub-reader.online/#)
            - sourcecode: https://github.com/vers-one/EpubReader
        - [Pressbooks](https://pressbooks.com/)
            - sourcecode: https://github.com/pressbooks/pressbooks
        - [Bookalope](https://bookalope.net/index.html)
            - sourcecode: https://github.com/bookalope
    
    - epub 架構: https://www.w3.org/publishing/epub32/epub-mediaoverlays.html 
        - [Calibre](https://calibre-ebook.com/)
            - sourcecode: https://github.com/kovidgoyal/calibre
            - [Calibre Web](https://github.com/janeczku/calibre-web)
            - requirements.txt
```
requirements.txt

APScheduler>=3.6.3,<3.11.0
Babel>=1.3,<3.0
Flask-Babel>=0.11.1,<4.1.0
Flask-Principal>=0.3.2,<0.5.1
Flask>=1.0.2,<3.1.0
iso-639>=0.4.5,<0.5.0
PyPDF>=3.15.6,<4.3.0
pytz>=2016.10
requests>=2.28.0,<2.32.0
SQLAlchemy>=1.3.0,<2.1.0
tornado>=6.3,<6.5
Wand>=0.4.4,<0.7.0
unidecode>=0.04.19,<1.4.0
lxml>=4.9.1,<5.3.0
flask-wtf>=0.14.2,<1.3.0
chardet>=3.0.0,<4.1.0
advocate>=1.0.0,<1.1.0
Flask-Limiter>=2.3.0,<3.9.0
regex>=2022.3.2,<2024.6.25
bleach>=6.0.0,<6.2.0
python-magic>=0.4.27,<0.5.0
flask-httpAuth>=4.4.0,<5.0.0
```

![image](https://hackmd.io/_uploads/Hk7fBaStR.png)

## Discard Object

### epub.js
CVE-2021-33040
```
描述
0.3.89 之前的 FuturePress EPub.js 中的 manager/views/iframe.js 允許 XSS。
```
issue
```
https://github.com/futurepress/epub.js/commit/ab4dd46408cce0324e1c67de4a3ba96b59e5012e
```

patch
```
// sandbox
		this.iframe.sandbox = "allow-same-origin";
		if (this.settings.allowScriptedContent && this.section.properties.indexOf("scripted") > -1) {
			this.iframe.sandbox += " allow-scripts"
		}
```

### Calibre web
CVE-2024-39123：Calibre-Web 跨站腳本 (XSS)
```
0.6.0 ~ 0.6.21
```

### Calibre
CVE 2023-46303的攻擊腳本
https://github.com/0x1717/ssrf-via-img
```
6.19.0之前的calibre中的ebooks/conversion/plugins/html_input.py中的link_to_local_path預設可以新增文檔根目錄以外的資源。
5.40.0可以攻擊成功
    攻擊思路應該是使用者打開calibre添加電子書的時候，添加了惡意製作的html造成路徑遍歷漏洞```
    
```

```
'''
                                                     _ooOoo
                                                    o8888888o
                                                    88" . "88 
                                                    (| -_- |)
                                                    O\  =  /O
                                                  ___/`---'\____
                                               .'  \\|     |//  `.
                                              /  \\|||  :  |||//  \
                                             /  _||||| -:- |||||_  \
                                             |   | \\\  -  /// |   |
                                             | \_|  ''\---/''  |   |
                                             \  .-\__       __/-.  /
                                           ___`. .'  /--.--\ `. . __
                                        ."" '<  `.___\_<|>_/__.'  >'"".
                                       | | :  `- \`.;`\ _ /`;.`/ - ` : | |
                                       \  \ `-.   \_ __\ /__ _/   .-` /  /
                                  ======`-.____`-.___\_____/___.-`____.-'======
                                                     `=---='
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                              佛祖保佑       永無BUG
'''
```
