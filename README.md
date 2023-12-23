# Machine-Learning_Final
52000691 - LeCongNghia. Các phương pháp Optimizer trong huấn huyện mô hình học máy. Continual Learning và Test Production

# <a name="_toc154182275"></a>**DANH MỤC CÁC CHỮ VIẾT TẮT**


|<p>Optimizer</p><p>Loss function</p><p>Neural network</p><p>Weight</p><p>Bias</p><p>Model</p><p>Epoch</p><p></p><p>Sample</p><p>Batch</p><p></p><p>Learning rate</p><p></p><p>Cost function / LossFunction</p><p>ML</p><p>Coutinual learning</p>|<p>Trình tối ưu</p><p>Hàm mất mát</p><p>Mạng nơ-ron</p><p>Trọng số</p><p>Độ lệnh</p><p>Mô hình</p><p>Số lần thuật toán chạy trên toàn bộ tập dữ liệu huấn luyện.</p><p>Một hàng của tập dữ liệu.</p><p>Biểu thị số lượng mẫu được lấy để cập nhật các tham số mô hình.</p><p>Là một tham số cung cấp cho mô hình thang đo về mức độ trọng số của mô hình cần được cập nhật. (Tốc độ học).</p><p>Được sử dụng để tính chi phí, là chênh lệch giữa giá trị dự đoán và giá trị thực tế.</p><p>Học máy</p><p>Học liên tục</p>|
| :- | :- |


1  # <a name="_toc154182276"></a>**CÁC PHƯƠNG PHÁP OPTIMIZER TRONG HUẤN LUYỆN MÔ HÌNH HỌC MÁY.**
   1. ## <a name="_toc154182277"></a>**Khái niệm về optimizer**
Optimizers (Trình tối ưu) là các thuật toán điều chỉnh các tham số của mô hình trong quá trình huấn luyện để giảm thiểu loss function (hàm mất mát). Là cơ sở để xây dựng mô hình neural network với mục đích "học" được các features (hay pattern) của dữ liệu đầu vào, từ đó có thể tìm một cặp weights và bias phù hợp để tối ưu hóa model.

Các trình tối ưu phổ biến bao gồm **Stochastic Gradient Descent (SGD), Adam và RMSprop**. Mỗi thuật toán tối ưu có các quy tắc cập nhật, tốc độ học và động lượng cụ thể để tìm các tham số mô hình tối ưu nhằm cải thiện hiệu suất.

1. ## <a name="_toc154182278"></a>**Tại sao phải sử dụng optimizer và ứng dụng của optimzer**
Chúng ta cho phép neural networks học từ bộ dữ liệu đầu vào. Nhưng vấn đề là "học" như thế nào? Cụ thể là weights và bias được tìm như thế nào! Đâu phải chỉ cần random (weights, bias) một số lần hữu hạn (cập nhật lặp đi lặp lại weights và biases) và hy vọng ở một bước nào đó ta có thể tìm được lời giải. Rõ ràng là không khả thi và lãng phí tài nguyên (Tốn thời gian và bộ nhớ)! Chúng ta phải tìm một thuật toán để cải thiện weight và bias theo từng bước, và đó là lý do các thuật toán optimizer ra đời.

Các thuật toán tối ưu là phương pháp tối ưu hoá giúp cải thiện hiệu suất của mô hình. Các thuật toán tối ưu hoá hoặc trình tối ưu này ảnh hưởng đến độ chính xác và tốc độ đào tạo của mô hình.

Trong khi huấn luyện mô hình tối ưu thì weights sẽ được sữa đổi trong các lần học và giảm thiểu hàm mất mát. Optimizer là một hàm hoặc thuật toán điều chỉnh các thuộc tính của neural network, chẳng hạn như trọng số và tốc độ học. Vì vậy nó giúp làm giảm tổn thất tổng thể và cải thiện độ chính xác. Vấn đề chọn weights phù hợp cho mô hình là một nhiệm vụ khó khăn, vì mô hình thường bao gồm hàng triệu tham số. Nó đặt ra nhu cầu chọn một thuật toán tối ưu hoá phù hợp cho từng ứng dụng.

Chúng ta có thể sử dụng các optimizer khác nhau trong mô hình học máy để thay đổi trọng số và tốc độ học. Tuy nhiên, việc chọn optimizer tốt nhất phụ thuộc vào ứng dụng. Khi mới bắt đầu, một ý nghĩ xâu xa xuất hiện trong đầu là chúng ta thử tất cả các khả năng và chọn một khả năng cho kết quả tốt nhất. Điều này ban đầu có thể ổn, nhưng khi xử lý hàng trăm gigabyte dữ liệu, ngay cả một lần học cũng có thể mất rất nhiều thời gian. Vì vậy, việc chọn ngẫu nhiên một thuật toán không kém gì việc đánh cược với thời gian quý giá của mình.

Trong bài báo cáo này, em sẽ đề cập đến nhiều optimizer khác nhau, chẳng hạn như: **Gradient Descent, Stochastic Gradient Descent, Stochastic Gradient Descent with Momentum, Mini-Batch Gradient Descent, Adagrad, RMSProp, AdaDelta và Adam**.

1. ## <a name="_toc154182279"></a>**Các thuật toán tối ưu, ưu và nhược điểm của các thuật toán**
   1. ### <a name="_toc154182280"></a>***Gradient Descent (GD)***
Trong các bài toán tối ưu, chúng ta thường tìm giá trị nhỏ nhất của một hàm số nào đó, mà hàm số đạt giá trị nhỏ nhất khi đạo hàm bằng 0. Nhưng đâu phải lúc nào đạo hàm hàm số cũng được, đối với các hàm số nhiều biến thì đạo hàm rất phức tạp, thậm chí là bất khả thi. Nên thay vào đó người ta tìm điểm gần với điểm cực tiểu nhất và xem đó là nghiệm bài toán. Gradient Descent tạm dịch là giảm dần độ dốc, nên hướng tiếp cận ở đây là chọn một nghiệm ngẫu nhiên cứ sau mỗi vòng lặp (hay epoch) thì cho nó tiến dần đến điểm cần tìm.

Chúng ta có thể nghĩ một ví dụ đơn giản là, hãy coi như chúng ta đang cầm một quả bóng nhỏ nằm trên đỉnh một cái bát. Khi chúng ta đánh vào quả bóng làm cho quả bóng rơi xuống, thì lúc này quả bóng sẽ đi theo hướng dốc nhất và cuối cùng là lắng xuống đáy bát. Một gradient cung cấp bóng theo hướng dốc nhất để đạt đến mức tối thiểu cục bộ là đáy bát.

Công thức: **X\_new = X – alpha \* f’(X)**

Trong đó: 

- **alpha**: là kích thước bước biểu thị khoảng cách di chuyển so với từng gradient với mỗi lần lặp.
- **Dấu** - : Ám chỉ ngược hướng đạo hàm.
- **Tại sao lại ngược hướng đạo hàm:** Ví dụ đối với hàm 

Gradient descent hoạt động như sau:

1. Nó bắt đầu với một số hệ số, xem chi phí của chúng và tìm kiếm giá trị chi phí thấp hơn giá trị hiện tại.
1. Nó di chuyển về phía trọng số thấp ưhon và cập nhật giá trị của các hệ số.
1. Quá trình lặp lại cho đến khi đạt được mức tối thiểu cục bộ. Mức tối thiểu cục bộ là một điểm vượt quá mức đó không thể tiếp tục nữa.

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.001.png)

<a name="_toc154182262"></a>Hình 1. Cách thức hoạt động của gradient descent

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.002.png)

<a name="_toc154182263"></a>Hình 2. Gradient descent cho hàm một biến

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.003.png)

<a name="_toc154182264"></a>Hình 3. Gradient descent cho hàm nhiều biến

Qua các hình trên ta thấy Gradient descent phụ thuộc vào nhiều yếu tố: như nếu chọn điểm X ban đầu khác nhau sẽ ảnh hưởng đến quá trình hội tụ; hoặc tốc độ học (learning rate) quá lớn hoặc quá nhỏ cũng ảnh hưởng: nếu tốc độ học quá nhỏ thì tốc độ hội tụ rất chậm ảnh hưởng đến quá trình training, còn tốc độ học quá lớn thì tiến nhanh tới đích sau vài vòng lặp tuy nhiên thuật toán không hội tụ, quanh quẩn quanh đích vì bước nhảy quá lớn.

Ưu điểm: 

- Thuật toán gradient descent cơ bản, dễ hiểu. 
- Thuật toán đã giải quyết được vấn đề tối ưu model neural network bằng cách cập nhật trọng số sau mỗi vòng lặp. 
- Hoạt động tốt đối với các hàm lồi.

Nhược điểm: 

- Phụ thuộc vào nghiệm khởi tạo ban đầu và learning rate. Ví dụ một hàm số có hai global minimum thì tùy thuộc vào hai điểm khởi tạo ban đầu sẽ cho ra hai nghiệm cuối cùng khác nhau. 
- Tốc độ học quá lớn sẽ khiến cho thuật toán không hội tụ, quanh quẩn bên đích vì bước nhảy quá lớn; hoặc tốc độ học nhỏ ảnh hưởng đến tốc độ training. 
- Việc tính toán độ dốc sẽ tốn kém nếu kích thước của dữ liệu lớn. 
- Nó không biết phải di chuyển bao xa dọc theo độ dốc với các hàm không lồi.

1. ### <a name="_toc154182281"></a>***Stochastic Gradient Descent (SGD)***
Ở cuối phần trước, chúng ta đã biết tại sao sử dụng phương pháp gradient descent trên dữ liệu lớn có thể không phải là lựa chọn tốt nhất. Việc tính toán đạo hàm trên toàn bộ dữ liệu qua mỗi vòng lặp trở nên cồng kềnh. Bên canh đó GD không phù hợp với **online Learning.** Nghĩa là khi dữ liệu cập nhật liên tục thì mỗi lần thêm dữ liệu ta phải tính lại đạo hàm trên toàn bộ dữ liệu. Điều này dẫn đến thời gian tính toán lâu, thuật toán không online nữa. Thay vào đó, mỗi lần thêm dữ liệu mới vào chỉ cần cập nhật trên một điểm dữ liệu đó thôi, phù hợp với online learning. Để giải quyết vấn đề, chúng ta có phương pháp **stochastic gradient descent** (giảm độ dốc ngẫu nhiên). Thuật ngữ stochastic (ngẫu nhiên) có nghĩa là tính ngẫu nhiên mà thuật toán dựa vào. Trong phương pháp stochastic gradient descent, thay vì lấy toàn bộ tập dữ liệu cho mỗi lần lặp, chúng ta sẽ chọn ngẫu nhiên các batch dữ liệu. Điều đó có nghĩa là chúng ta chỉ lấy một vài mẫu từ tập dữ liệu.

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.004.png)

<a name="_toc154182265"></a>Hình 4. Mô hình học Stochastic Gradient descent và Gradient descent

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.005.png)

Quy trình đầu tiên là chọn các tham số ban đầu w và tốc độ học n. Sau đó xoá trộn ngẫu nhiên dữ liệu ở mỗi lần lặp để đạt mức tối thiểu gần đúng.

Vì chúng ta không sử dụng toàn bộ tập dữ liệu mà sử dụng các lô (batch) của tập dữ liệu cho mỗi lần lặp, nên đường đi của thuật toán có nhiều nhiễu so với thuật toán GD. Do đó, SGD sử dụng số lần lặp cao hơn để đạt cực tiểu cục bộ. Do số lần lặp tăng lên nên thời gian tính toán tổng thể tăng lên. Nhưng ngay cả sau khi tăng số lần lặp, chi phí tính toán vẫn thấp hơn so với trình tối ưu hóa GD. Vì vậy, kết luận là nếu dữ liệu rất lớn và thời gian tính toán là một yếu tố thiết yếu, thì việc SGD nên được ưu tiên hơn thuật toán GD hàng loạt.

Ưu điểm: Thuật toán giải quyết được đối với cơ cở dữ liệu lớn mà GD không làm được. Thuật toán tối ưu này hiện nay vẫn hay được sử dụng.

Nhược điềm: Thuật toán vẫn chưa giải quyết đươc hai nhược điểm lớn của gradient descent (learning rate, điểm dữ liệu ban đầu). Vì vậy ta phải kết hợp SGD với một số thuật toán khác như: Momentum, AdaGrad, …Các thuật toán này sẽ được em trình bày ở phần sau.

1. ### <a name="_toc154182282"></a>***Stochastic Gradient Descent with Momentum***
Như chúng ta đã thảo luận ở phần trước, chúng ta đã biết rằng thuật toán Stochastic Gradient descent có đường đi nhiễu hơn nhiều so với thuật toán Gradient descent. Vì lý do này, nó đòi hỏi số lần lặp đáng kể hơn để đạt được mức tối thiểu tối ưu và do đó thời gian tính toán rất chậm. Để khắc phục vấn đề, chúng ta sử dụng phương pháp Stochastic Gradient descent với thuật toán động lượng (Momentum).

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.006.png)

<a name="_toc154182266"></a>Hình 5. Ý tưởng của thuật toán GD và GD with Momentum

Để giải thích được Stochastic Gradient descent with Momentum thì trước tiên ta nên nhìn dưới góc độ vật lí: Như hình b phía trên, nếu ta thả hai viên bi tại hai điểm khác nhau A và B thì viên bị A sẽ trượt xuống điểm C còn viên bi B sẽ trượt xuống điểm D, nhưng ta lại không mong muốn viên bi B sẽ dừng ở điểm D (local minimum) mà sẽ tiếp tục lăn tới điểm C (global minimum). Để thực hiện được điều đó ta phải cấp cho viên bi B một vận tốc ban đầu đủ lớn để nó có thể vượt qua điểm E để tiến tới điểm C. Dựa vào ý tưởng này người ta xây dựng nên thuật toán Momentum (tức là theo đà tiến tới).

Momentum làm gì giúp hàm mất mát hội tụ nhanh hơn. Giảm dần độ dốc ngẫu nhiên dao động giữa một trong hai hướng của độ dốc và cập nhật trọng số tương ứng. Tuy nhiên, việc thêm một phần bản cập nhật trước đó vào bản cập nhật hiện tại sẽ giúp quá trình diễn ra nhanh hơn một chút. Một điều cần nhớ khi sử dụng thuật toán này là tốc độ học sẽ giảm với số hạng momentum cao.

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.007.png)

<a name="_toc154182267"></a>Hình 6. Giảm số lần lặp của mô hình khi sử dụng SGD with Momentum

Trong hình ảnh trên, phần bên trái hiển thị biểu đồ hội tụ của thuật toán giảm độ dốc ngẫu nhiên. Đồng thời, phía bên phải hiển thị SGD có động lượng. Từ hình ảnh, bạn có thể so sánh đường đi được cả hai thuật toán chọn và nhận ra rằng việc sử dụng động lượng giúp đạt được sự hội tụ trong thời gian ngắn hơn. Bạn có thể đang nghĩ đến việc sử dụng động lượng và tốc độ học tập lớn để khiến quá trình diễn ra nhanh hơn nữa. Nhưng hãy nhớ rằng khi tăng đà, khả năng vượt qua mức tối thiểu tối ưu cũng tăng lên. Điều này có thể dẫn đến độ chính xác kém và thậm chí nhiều dao động hơn.

Nhìn dưới góc độ toán học ta có công thực Momentum:

**X\_new = X\_old – (gama.v + n.f’(X))**

Trong đó:

- X\_new: Toạ độ mới
- X\_old: Toạ độ cũ
- Gama: Parameter, thường = 0.9
- n: Tốc độ học (Learning rate)
- f’(X): Đạo hàm của hàm f (gradient)

SGD with momentum dù mất nhiều vòng lặp hơn để dừng lại nhưng nghiệm tiến tới global minimum vì Điểm chạy X sẽ dao động qua lại quanh điểm đích đó trước khi dừng lại.

Ưu điểm: Thuật toán tối ưu giải quyết được vấn đề GD không tiến được tới điểm global minimum mà chỉ dừng lại ở local minimum.

Nhược điểm: Tuy momentum giúp hòn bi vượt dốc tiến tới điểm đích, tuy nhiên khi tới gần đích, nó vẫn mất khá nhiều thời gian giao động qua lại trước khi dừng hẳn, điều này được giải thích vì nó đang có đà.

1. ### <a name="_toc154182283"></a>***Mini Batch Gradient Descent***
Giảm độ dốc theo lô nhỏ (mini-batch) là một biến thể của thuật toán giảm độ dốc nhằm chia tập dữ liệu huấn luyện thành các mini-batch được sử dụng để tính toán lỗi mô hình và cập nhật hệ số mô hình.

Trong biến thể giảm độ dốc này, thay vì lấy tất cả dữ liệu huấn luyện, chỉ một tập hợp con của tập dữ liệu được sử dụng để tính hàm mất mát. Vì chúng ta đang sử dụng một loạt dữ liệu thay vì lấy toàn bộ tập dữ liệu nên cần ít lần lặp hơn. Đó là lý do tại sao thuật toán Mini Batch Gradient Descent nhanh hơn cả thuật toán Stochastic Gradient Descent và thuật toán Batch Gradient Descent. Thuật toán này hiệu quả và mạnh mẽ hơn các biến thể Gradient Descent trước đó. Vì thuật toán sử dụng tính năng bó (batching), tất cả dữ liệu huấn luyện không cần phải được tải vào bộ nhớ, do đó giúp quá trình thực hiện hiệu quả hơn. Hơn nữa, hàm chi phí trong thuật toán Mini Batch Gradient Descent ồn (noisier) hơn so với thuật toán Batch Gradient Descent nhưng mượt mà hơn so với thuật toán Stochastic Gradient Descent. Do đó, Mini Batch Gradient Descent là lý tưởng và mang lại sự cân bằng tốt giữa tốc độ và độ chính xác.

**Mini Batch Gradient Descent** nhằm tìm ra sự cân bằng giữa độ mạnh mẽ của việc Stochastic Gradient Descent và hiệu quả của việc Batch Gradient Descent. Đây là cách triển khai phổ biến nhất của việc Gradient Descent được sử dụng trong lĩnh vực học sâu.

Ưu điểm: 

- Tần suất cập nhật mô hình cao hơn so với Batch Gradient Descent, cho phép hội tụ mạnh mẽ hơn, tránh mức tối thiểu cục bộ.
- The batched updates provide a computationally more efficient process than stochastic gradient descent

Nhược điểm: 

- Bất chấp tất cả những điều đó, thuật toán **Mini Batch Gradient Descent** cũng có một số nhược điểm. Nó cần một siêu tham số có "mini-batch-size", cần được điều chỉnh để đạt được độ chính xác cần thiết. Mặc dù vậy, batch-size là 32 được coi là phù hợp với hầu hết mọi trường hợp. Ngoài ra, trong một số trường hợp, nó dẫn đến độ chính xác cuối cùng kém. Do đó, cần phải tăng cường tìm kiếm các lựa chọn thay thế khác. 

1. ### <a name="_toc154182284"></a>***Adagrad (Adaptive Gradient Descent)***
Thuật toán giảm độ dốc thích ứng hơi khác so với các thuật toán giảm độ dốc khác. Điều này là do nó sử dụng các tốc độ học (learning rate) khác nhau cho mỗi lần lặp. Hay nói cách khác Adagrad coi learning rate là một tham số, tức là Adagrad sẽ cho learning rate biến thiên sau mỗi thời điểm t. Sự thay đổi learning rate phụ thuộc vào sự khác biệt của các tham số trong quá trình huấn luyện. Các tham số càng thay đổi thì learning rate càng thay đổi nhỏ. Việc sửa đổi này rất có lợi vì các bộ dữ liệu trong thế giới thực chứa các tính năng thưa thớt cũng như dày đặc. Vì vậy, thật không công bằng khi có cùng một giá trị learning rate cho tất cả các đặc tính. Thuật toán Adagrad sử dụng công thức dưới đây để cập nhật trọng số.

W<sub>t+1</sub> = W<sub>t</sub> – n’<sub>t</sub> \* g(t)

n’<sub>t</sub> = nαt+ ϵ

Trong đó: 

- alpha(t): biểu thị tốc độ học khác nhau ở mỗi lần lặp. (là ma trận chéo mà mỗi phần tử trên đường chéo (i, i) là bình phương của đạo hàm vectơ tham số tại thời điểm t.)
- n: là hằng số.
- ∈: là giá trị dương nhỏ để biểu thức tránh chia cho 0
- g(t): Gradient tại thời điểm t

Ưu điểm: Nó loại bỏ sự cần thiết phải sửa đổi learning rate một cách thủ công. Nó đáng tin cậy hơn các thuật toán GD và các biến thể của chúng và đạt được sự hội tụ ở tốc độ cao hơn.

Nhược điểm: Nó làm giảm learning rate một cách mạnh mẽ và đơn điệu. Có thể có một thời điểm khi learning rate trở nên cực kỳ nhỏ. Điều này là do gradient bình phương ở mẫu số tiếp tục tích lũy và do đó phần mẫu số tiếp tục tăng. Do learning rate nhỏ, mô hình cuối cùng không thể thu được nhiều kiến ​​thức hơn và do đó độ chính xác của mô hình bị ảnh hưởng hoặc có thể trở nên đóng băng.

1. ### <a name="_toc154182285"></a>***RMS Prop (Root Mean Square)***
RMSProp là một kỹ thuật tối ưu hóa dựa trên độ dốc được sử dụng trong đào tạo neural networks và được phát triển như một kỹ thuật stochastic cho việc học theo đợt nhỏ (mini-batch learning). Độ dốc của các hàm rất phức tạp như neural networks có xu hướng biến mất hoặc bùng nổ khi dữ liệu truyền qua hàm.

RMSProp giải quyết vấn đề trên bằng cách sử dụng đường trung bình động của các gradient bình phương để chuẩn hóa gradient. Việc chuẩn hóa này cân bằng kích thước bước (momentum), giảm bước cho các gradient lớn để tránh bùng nổ và tăng bước cho các gradient nhỏ để tránh biến mất.

Thuật toán giữ giá trị trung bình động của các gradient bình phương cho mọi trọng số và chia gradient cho căn bậc hai của tổng bình phương của gradient g<sub>t</sub>.

S<sub>t</sub> = γSt-1+ 1-γgt2

W<sub>t</sub> = Wt-1- nSt + ∈ \* gt

Trong đó:

S<sub>t</sub>: Tổng bình phương gradient g<sub>t</sub> tại thời điểm t

n: Tốc độ học (hằng số)

∈: Tham số dương nhỏ để tránh trường hợp biểu thức chia cho 0.

Ưu điểm: 

- Giải quyết được vấn đề tốc độ học giảm dần của Adagrad (vấn đề tốc độ học giảm dần theo thời gian sẽ khiến việc training chậm dần, có thể dẫn tới bị đóng băng).
- Sự hội tụ nhanh hơn momentum.
- Có thể xử lý các đối tượng ngẫu nhiên (stochastic) rất tốt, giúp nó có thể áp dụng cho việc học theo đợt nhỏ.

Nhược điểm: 

- Learning rate phải được xác định theo cách thủ công và giá trị được đề xuất không hoạt động cho mọi ứng dụng.
- Kết quả nghiệm chỉ là local minimum chứ không đạt được global minimum như Momentum. Vì vậy người ta sẽ kết hợp cả 2 thuật toán Momentum với RMSprop cho ra 1 thuật toán tối ưu Adam. Chúng ta sẽ trình bày nó trong phần sau.

1. ### <a name="_toc154182286"></a>***AdaDelta***
AdaDelta có thể được coi là phiên bản mạnh mẽ hơn của trình tối ưu hóa AdaGrad. Nó dựa trên phương pháp học tập thích ứng và được thiết kế để giải quyết những hạn chế đáng kể của trình tối ưu hóa hỗ trợ AdaGrad và RMS. Vấn đề chính với hai trình tối ưu hóa ở trên là tốc độ học ban đầu phải được xác định theo cách thủ công. Một vấn đề khác là tốc độ học giảm dần, tại một thời điểm nào đó nó trở nên cực kỳ nhỏ. Do đó, sau một số lần lặp nhất định, mô hình không thể học được kiến ​​thức mới nữa.

Điểm khác biệt chính là Adadelta giảm mức độ mà tốc độ học sẽ thay đổi với các tọa độ. Hơn nữa, Adadelta thường được biết đến là thuật toán không sử dụng tốc độ học vì nó dựa trên chính lượng thay đổi hiện tại để căn chỉnh lượng thay đổi trong tương lai.

Adadelta sử dụng hai biến trạng thái, S<sub>t</sub> để lưu trữ trung bình rò rỉ mô-men bậc hai của gradient và ΔX<sub>t</sub> để lưu trữ trung bình rò rỉ mô-men bậc hai của lượng thay đổi của các tham số trong mô hình.

S<sub>t = ρSt-1+ 1-ρgt2</sub>

g’<sub>t</sub> = ∆Xt-1 + ∈St + ∈ \* gt

W<sub>t</sub> = W<sub>t – 1</sub> – g’<sub>t</sub>

∆Wt= ρ∆Wt-1+1-ρWt2

Điểm khác biệt so với trước là ta thực hiện các bước cập nhật với gradient g’<sub>t</sub> được tái tỉ lệ bằng cách lấy căn bậc hai thương của trung bình tốc độ thay đổi bình phương và trung bình mô-men bậc hai của gradient.

Ưu đểm: 

- AdaDelta tự điều chỉnh learning rate mà không cần phải thiết lập một learning rate cố định. Điều này giúp giảm bớt công đoạn tinh chỉnh tham số quan trọng trong quá trình đào tạo.
- AdaDelta giảm thiểu vấn đề của vanishing hoặc exploding gradient bằng cách sử dụng trung bình động của bình phương gradient.
- AdaDelta có khả năng làm việc hiệu quả trên dữ liệu không đồng nhất hoặc có độ biến động lớn.
- AdaDelta giảm thiểu ảnh hưởng của nhiễu (noise) trong quá trình cập nhật tham số.

Nhược điểm:

- AdaDelta yêu cầu tính toán trung bình động của bình phương gradient và tham số cập nhật, điều này làm tăng độ phức tạp tính toán so với một số thuật toán khác.
- AdaDelta không phải là một lựa chọn tốt cho tất cả các loại mô hình và tập dữ liệu. Có những trường hợp nó không thể hiệu quả như một số bài toán cụ thể hoặc khi mô hình yêu cầu sự cập nhật chính xác của learning rate.

1. ### <a name="_toc154182287"></a>***Adam***
Trình tối ưu hóa Adam, viết tắt của Adaptive Moment Estimation optimizer (Trình tối ưu hóa ước tính thời điểm thích ứng). Nó là một phần mở rộng của thuật toán Stochastic Gradient Descent (SGD) và được thiết kế để cập nhật trọng số của mạng nơ-ron trong quá trình đào tạo.

Cái tên "Adam" có nguồn gốc từ "ước tính thời điểm thích ứng", nêu bật khả năng điều chỉnh thích ứng learning rate cho từng trọng lượng mạng riêng lẻ. Không giống như SGD duy trì một tốc độ học duy nhất trong suốt quá trình đào tạo, trình tối ưu hóa Adam tự động tính toán tốc độ học của từng cá nhân dựa trên độ dốc trong quá khứ và momentum thứ hai của chúng.

Những người tạo ra trình tối ưu hóa Adam đã kết hợp các tính năng có lợi của các thuật toán tối ưu hóa khác như AdaGrad và RMSProp. Tương tự như RMSProp, trình tối ưu hóa Adam xem xét momentum thứ hai của độ dốc, nhưng không giống như RMSProp, nó tính toán phương sai không tập trung của độ dốc (không trừ giá trị trung bình).

Nếu giải thích theo hiện tượng vật lí thì Momentum giống như 1 quả cầu lao xuống dốc, còn Adam như 1 quả cầu rất nặng có ma sát, vì vậy nó dễ dàng vượt qua local minimum tới global minimum và khi tới global minimum nó không mất nhiều thời gian dao động qua lại quanh đích vì nó có ma sát nên dễ dừng lại hơn.

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.008.png)

<a name="_toc154182268"></a>Hình 7. Ví dụ minh hoạ về cách hoạt động của Adam

Tóm lại, Adam Optimator là một thuật toán tối ưu hóa mở rộng SGD bằng cách điều chỉnh tự động tốc độ học dựa trên các trọng số riêng lẻ. Nó kết hợp các tính năng của Adagrad và RMSProp để cung cấp các bản cập nhật hiệu quả và thích ứng cho các trọng số mạng trong quá trình huấn luyện.

Trình tối ưu hóa adam có một số lợi ích nên nó được sử dụng rộng rãi. Nó được điều chỉnh làm chuẩn cho các bài viết về deep learning và được đề xuất làm thuật toán tối ưu hóa mặc định. Hơn nữa, thuật toán này dễ thực hiện, có thời gian chạy nhanh hơn, yêu cầu bộ nhớ thấp và yêu cầu điều chỉnh ít hơn bất kỳ thuật toán tối ưu hóa nào khác.

g<sub>t</sub> = ∇fθt-1= δLδwt

m<sub>t</sub> = β1mt-1+1-β1gt

v<sub>t</sub> = β2vt-1+1-β2gt2

w<sub>t</sub> = wt-1-a mt/vt+ ∈

g<sub>t</sub>: Giá trị đạo hàm của tham số mô hình tại thời điểm t

m<sub>t</sub>: Được xem như trong lượng của một vật

v<sub>t</sub>: Được xem như vận tốc của một vật

w<sub>t</sub>: Tróng số tại thời điểm t và được tính theo lượng ma sát của vật theo m, v

Công thức trên thể hiện hoạt động của bộ tối ưu hóa Adam. Ở đây β1 và ​​​​β2 biểu thị tốc độ phân rã trung bình của gradients.

Nếu trình tối ưu hóa Adam sử dụng các đặc tính tốt của tất cả các thuật toán và là trình tối ưu hóa tốt nhất hiện có thì tại sao chúng ta không nên sử dụng Adam trong mọi ứng dụng? Và nhu cầu tìm hiểu sâu về các thuật toán khác là gì? Điều này là do ngay cả Adam cũng có một số nhược điểm. Nó có xu hướng tập trung vào thời gian tính toán nhanh hơn, trong khi các thuật toán như SGD tập trung vào các điểm dữ liệu. Đó là lý do tại sao các thuật toán như SGD khái quát hóa dữ liệu theo cách tốt hơn nhưng lại phải trả giá bằng tốc độ tính toán thấp. Vì vậy, các thuật toán tối ưu hóa có thể được chọn phù hợp tùy thuộc vào yêu cầu và loại dữ liệu.

1. ## <a name="_toc154182288"></a>**Khi nào áp dụng optimizer nào**
Còn có rất nhiều thuật toán tối ưu như Nesterov (NAG), Adadelta, Nadam, ... Hiện nay optimizers hay được sử dụng nhất là 'Adam'.

Vì vậy, bây giờ bạn nên sử dụng trình tối ưu hóa nào? Nếu dữ liệu đầu vào thưa thớt thì chúng ta có thể đạt được kết quả tốt nhất bằng cách sử dụng một trong các phương pháp tốc độ học thích ứng. Một lợi ích nữa là bạn sẽ không cần điều chỉnh tốc độ học nhưng có thể đạt được kết quả tốt nhất với giá trị mặc định.

Đối với tất cả các trình tối ưu hóa hiện nay, minibatch luôn được sử dụng. Mini Batch Gradient Descent đã giải quyết được vấn đề về hiệu suất và có ít tiếng ồn hơn, trong khi momentum giảm tiếng ồn để mang lại hiệu ứng làm mượt. Vấn đề giảm learning rate trong neural network sâu hơn đã được RMSProp giải quyết và Adam bằng cách sử dụng cả momentum và RMSProp đã trở thành trình tối ưu hóa tốt nhất hiện nay.

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.009.png)

<a name="_toc154182269"></a>Hình 8. Tổng quan về các thuật toán tối ưu


1  # <a name="_toc154182289"></a>**CONTINUAL LEARNING VÀ TEST PRODUCTION**
   1. ## <a name="_toc154182290"></a>**Continual Learning**
Học tập liên tục, còn được gọi là học tập suốt đời hoặc học tập tăng dần, là một mô hình học máy tập trung vào các mô hình đào tạo để tiếp thu kiến ​​thức mới và thích ứng với việc thay đổi dữ liệu theo thời gian. Ngược lại với học máy truyền thống, trong đó các mô hình thường được đào tạo trên các tập dữ liệu cố định và giả định rằng việc phân phối dữ liệu không đổi, học liên tục được thiết kế để xử lý các phân phối dữ liệu đang phát triển và liên tục học hỏi từ dữ liệu mới trong khi vẫn giữ được kiến ​​thức từ những trải nghiệm trước đó. Điều này đặc biệt quan trọng trong các trường hợp dữ liệu không cố định, nghĩa là nó thay đổi theo thời gian.

Là ý tưởng cập nhật mô hình của chúng ta khi có dữ liệu mới, điều này làm cho mô hình của chúng ta theo kịp các phân phối dữ liệu hiện tại.

Continual learning thường bị hiểu sai:

- Continual learning KHÔNG đề cập đến một lớp thuật toán ML đặc biệt cho phép cập nhật mô hình dần dần khi có mọi điểm dữ liệu mới. Ví dụ về lớp thuật toán đặc biệt này là cập nhật bayesian tuần tự *(sequential bayesian updating)* và phân loại KNN *(KNN classifiers)*. Lớp thuật toán này nhỏ và đôi khi được gọi là "thuật toán học trực tuyến" *(online learning algorithms).*
- Khái niệm Học liên tục có thể được áp dụng cho bất kỳ thuật toán ML được giám sát nào. Không, chỉ là một lớp học đặc biệt.
- Continual learning KHÔNG có nghĩa là bắt đầu công việc đào tạo lại mỗi khi có mẫu dữ liệu mới. Trên thực tế, điều này rất nguy hiểm vì nó khiến neural network dễ bị lãng quên một cách thảm khốc.

Một giả định chính của học máy là dữ liệu huấn luyện cho một mô hình được lấy từ cùng một miền và có cùng đặc điểm (ví dụ: tính năng đầu vào, phân phối) như dữ liệu thử nghiệm. Điều này đảm bảo rằng các mô hình có thể khái quát hóa tốt với dữ liệu mới chưa được nhìn thấy. Tuy nhiên, trong nhiều tình huống thực tế thì điều này không xảy ra và dữ liệu huấn luyện cho một nhiệm vụ học tập chỉ khả dụng trong một thời gian nhất định. Trong những trường hợp như vậy, một mô hình mới sẽ phải được đào tạo từ đầu về dữ liệu mới cho mỗi nhiệm vụ mới. Học liên tục (Continual Learning) là một mô hình học máy giải quyết vấn đề này và xử lý các mô hình học máy đào tạo theo thời gian sao cho các mô hình này vừa có thể tiếp thu kiến ​​thức cho các nhiệm vụ mới, vừa giữ lại kiến ​​thức từ các nhiệm vụ được đào tạo trước. Tuy nhiên, khi đào tạo neural network bằng phương pháp học chuyển giao, chúng có thể bị quên lãng nghiêm trọng. Ngay sau khi chúng được huấn luyện cho các nhiệm vụ diễn ra tuần tự, hiệu suất của chúng đối với các nhiệm vụ đã học trước đó sẽ giảm do thay đổi các tham số tương ứng của trọng số mạng.

Ngược lại, các phương pháp học liên tục giải quyết vấn đề này và cố gắng tìm ra sự cân bằng giữa tính ổn định và tính linh hoạt của các tham số mạng khi đào tạo các nhiệm vụ mới. Các phương pháp tiên tiến nhất để học liên tục có thể được chia đại khái thành ba loại: chiến lược diễn tập dựa trên trí nhớ (memory-based rehearsal strategies), kiến ​​trúc động (dynamic architectures) và chiến lược chính quy hóa (regularization strategies).

1. ### <a name="_toc154182291"></a>***Why continual learning?***
Lý do cơ bản là giúp mô hình theo kịp sự thay đổi phân phối dữ liệu. Có một số trường hợp sử dụng trong đó việc thích ứng nhanh chóng với việc thay đổi phân phối là rất quan trọng. Dưới đây là một số ví dụ:

- **Các trường hợp sử dụng có thể xảy ra những thay đổi nhanh chóng và bất ngờ**: Các trường hợp sử dụng như chia sẻ chuyến đi phải tuân theo điều này. Ví dụ: có thể có một buổi hòa nhạc ở một khu vực ngẫu nhiên vào Thứ Hai ngẫu nhiên và "mô hình ML định giá vào Thứ Hai" có thể không được trang bị tốt để xử lý buổi hòa nhạc đó.
- **Các trường hợp sử dụng trong đó không thể lấy dữ liệu huấn luyện cho một sự kiện cụ thể**. Một ví dụ cho điều này là các mô hình thương mại điện tử trong dịp Black Friday hay một số sự kiện sale khác chưa từng được thử trước đây. Rất khó để thu thập dữ liệu lịch sử để dự đoán hành vi của người dùng trong ngày Black Friday, vì vậy mô hình của bạn phải thích ứng suốt cả ngày.
- **Các trường hợp sử dụng nhạy cảm với vấn đề khởi động nguội**. Sự cố này xảy ra khi mô hình của chúng ta phải đưa ra dự đoán cho người dùng mới (hoặc đã đăng xuất) không có dữ liệu lịch sử (hoặc dữ liệu đã lỗi thời). Nếu chúng ta không điều chỉnh mô hình của mình ngay khi nhận được một số dữ liệu từ người dùng đó, chúng ta sẽ không thể đề xuất những điều liên quan cho người dùng đó.

1. ### <a name="_toc154182292"></a>***Các phương pháp tiên tiến của continual learning:***
**Phương pháp diễn tập sử dụng bộ nhớ có kích thước cố định** để lưu trữ các mẫu dữ liệu từ các tác vụ đã được đào tạo trước đó. Sau đó, các mẫu này sẽ được xem lại trong quá trình đào tạo các nhiệm vụ mới nhằm giảm thiểu tình trạng quên nghiêm trọng. Ví dụ, *Rebuffi và cộng sự. (2017)* lưu giữ trí nhớ theo từng giai đoạn với các mẫu đại diện cho từng nhiệm vụ. Khi đào tạo các nhiệm vụ mới, họ tính toán tổn thất chưng cất bổ sung để ngăn các dự đoán của mạng đối với các mẫu này thay đổi đáng kể. Ngược lại, *Lopez-Paz và Ranzato (2017)* sử dụng bộ nhớ để tính toán độ dốc của mạng cho các tác vụ trước đó. Sau đó, họ hình thành việc học một nhiệm vụ mới như một bài toán tối ưu hóa kép cho phép các gradient được tính toán giảm thiểu cả tổn thất mới và tổn thất trước đó.

**Các phương pháp tiếp cận với kiến ​​trúc động** sẽ thay đổi kiến ​​trúc của mạng khi đào tạo cho các nhiệm vụ mới. Thông thường, chúng tự động mở rộng dung lượng của mạng để học các mẫu mới mà không có xung đột. *Parisi và cộng sự. (2017)* sử dụng phương pháp tiếp cận ngày càng tăng khi được yêu cầu (GWR) để huấn luyện các mạng nơ-ron tự tổ chức định kỳ được mở rộng theo cấp bậc cho các nhiệm vụ mới. *Rus và cộng sự. (2016)* đề xuất một mạng lưới thần kinh đang phát triển dần dần, với mạng được mở rộng thêm một cột khi được đào tạo về một nhiệm vụ mới. Bằng cách đóng băng các cột trước đó và sử dụng các kết nối bên với cột mới, cả việc quên nghiêm trọng đều được ngăn chặn và kiến ​​thức đã học trước đó được tái sử dụng cho nhiệm vụ hiện tại.

**Các chiến lược chính quy hóa** giúp giảm thiểu tình trạng quên neural network một cách thảm khốc bằng cách hạn chế cập nhật tham số mạng trong khi đào tạo các nhiệm vụ mới. Trong hợp nhất trọng lượng đàn hồi (EWC) của *Kirkpatrick et al. (2017),* điều này được hiện thực hóa bằng cách xử phạt những thay đổi của các tham số quan trọng đối với các nhiệm vụ trước đó. Tầm quan trọng của tham số được ước tính thông qua mật độ xác suất sử dụng ma trận thông tin ngư dân. Do đó, EWC phù hợp cho các nhiệm vụ phân loại liên tục. Ngược lại, phương pháp tiếp nhận biết bộ nhớ do *Aljundi et al đề xuất. (2018, 2019)* thể hiện tầm quan trọng của các tham số mạng bằng độ nhạy của đầu ra mạng đối với những thay đổi của tham số. Bằng cách kết hợp các thay đổi của các tham số quan trọng vào hàm mất mát, chủ yếu các trọng số mạng đó được điều chỉnh cho phù hợp với một nhiệm vụ mới chưa quan trọng. 

**Progressive Neural Networks (PNNs):** 

- Mạng thần kinh tiến bộ (PNN) được thiết kế để học dần dần các nhiệm vụ mới trong khi vẫn duy trì kiến ​​thức về các nhiệm vụ đã biết trước đó. Ý tưởng chính đằng sau PNN là mở rộng công suất của mô hình khi có nhiệm vụ mới. Thay vì sử dụng một mạng nơron đơn lẻ, PNN sử dụng một tập hợp mạng. Mỗi mạng trong nhóm được dành riêng cho một nhiệm vụ cụ thể. Một mạng lưới thần kinh mới được thêm vào tập hợp khi một nhiệm vụ mới được đưa ra. Sau đó, mô hình kết hợp đầu ra của tất cả các mạng để đưa ra dự đoán.
- Lợi ích của PNN là chúng ngăn chặn sự quên lãng nghiêm trọng bằng cách cô lập kiến ​​thức liên quan đến từng nhiệm vụ trong các mạng chuyên dụng. Tuy nhiên, tập hợp có thể trở nên lớn khi có nhiều nhiệm vụ được học, điều này có thể dẫn đến độ phức tạp tính toán tăng lên.

**Learning without Forgetting (LwF):**

- Học mà không quên (LwF) là một phương pháp tiếp cận tận dụng việc chắt lọc kiến ​​thức để giải quyết tình trạng quên lãng nghiêm trọng. Ý tưởng là sử dụng mô hình được đào tạo trước làm mạng giáo viên và mạng lưới thần kinh mới làm học sinh. Khi học một nhiệm vụ mới, mạng học sinh được huấn luyện để bắt chước dự đoán của giáo viên về dữ liệu cũ và mới. Quá trình này giúp mạng học sinh ghi nhớ được kiến ​​thức từ các nhiệm vụ trước đó.
- LwF có hiệu quả về mặt tính toán vì nó không yêu cầu duy trì một tập hợp mạng lớn. Nó đặc biệt thành công trong các tình huống trong đó việc tinh chỉnh mô hình được đào tạo trước là thuận lợi.

**iCaRL (Incremental Classifier and Representation Learning):**

- iCaRL là một thuật toán được thiết kế cho các nhiệm vụ học tập liên tục liên quan đến phân loại. Nó kết hợp các chiến lược để học biểu diễn tính năng và lưu trữ mẫu dành riêng cho lớp. Mô hình duy trì một tập hợp các mẫu (mẫu đại diện) từ mỗi lớp đã học trước đó. Khi các lớp mới được giới thiệu, iCaRL sử dụng các mẫu này để lưu giữ kiến ​​thức về các lớp cũ.
- iCaRL rất phù hợp cho các nhiệm vụ cần quan tâm đến sự mất cân bằng giữa các lớp vì nó đảm bảo rằng mô hình giữ lại kiến ​​thức của cả lớp cũ và lớp mới trong khi thích ứng với dữ liệu mới.

**Meta-Learning Approaches:**

- Siêu học tập bao gồm các mô hình đào tạo để học hiệu quả và cũng đã được áp dụng cho việc học tập liên tục. Trong siêu học tập để học liên tục, các mô hình được đào tạo về nhiều nhiệm vụ khác nhau để có được chiến lược khởi tạo hoặc học tập tốt nhằm thích ứng nhanh chóng với các nhiệm vụ mới.
- Các kỹ thuật siêu học đã cho thấy hứa hẹn trong việc giảm thiểu tình trạng quên lãng nghiêm trọng bằng cách trang bị cho các mô hình một điểm khởi đầu vững chắc để học các nhiệm vụ mới.

1. ### <a name="_toc154182293"></a>***Khái niệm: Đào tạo lại không trạng thái và Đào tạo có trạng thái***
![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.010.png)

<a name="_toc154182270"></a>Hình 9. Đào tạo không trạng thái so với đào tạo trạng thái
1. #### Stateless retraining (Đào tạo lại không trạng thái)
Đào tạo lại mô hình từ đầu mỗi lần lặp, sử dụng trọng số được khởi tạo ngẫu nhiên và dữ liệu mới hơn.

- Có thể có một số trùng lặp với dữ liệu đã được sử dụng để huấn luyện phiên bản mô hình trước đó.
- Hầu hết các công ty bắt đầu **continual learning** bằng cách sử dụng đào tạo lại không trạng thái.
  1. #### Stateful training (aka fine-tuning, incremental learning) (Đào tạo có trạng thái)
Khởi tạo mô hình với các trọng số từ vòng huấn luyện trước và tiếp tục huấn luyện bằng cách sử dụng dữ liệu mới chưa thấy.

- Cho phép mô hình của mình cập nhật với lượng dữ liệu ít hơn đáng kể.
- Cho phép mô hình của mình hội tụ nhanh hơn và sử dụng ít năng lượng tính toán hơn. (Một số công ty đã báo cáo giảm 45% sức mạnh tính toán).
- Về mặt lý thuyết, nó có thể tránh việc lưu trữ dữ liệu hoàn toàn sau khi dữ liệu đã được sử dụng để đào tạo (và để lại một khoảng thời gian an toàn). Về mặt lý thuyết, điều này giúp loại bỏ những lo ngại về quyền riêng tư dữ liệu. (Trên thực tế, hầu hết các công ty đều có thói quen theo dõi mọi thứ và không muốn vứt bỏ dữ liệu ngay cả khi nó không còn cần thiết nữa).
- Sau khi cơ sở hạ tầng của chúng ta được thiết lập chính xác, việc thay đổi từ **stateless retraning** sang **stateful training** sẽ trở thành một nút nhấn.
- **Lặp lại mô hình và lặp lại dữ liệu**: Stateful training chủ yếu được sử dụng để kết hợp dữ liệu mới *vào kiến ​​trúc mô hình cố định và hiện có* (tức là lặp lại dữ liệu). Nếu muốn thay đổi các tính năng hoặc kiến ​​trúc của mô hình, chúng ta sẽ cần phải thực hiện quá trình stateless retraining lần đầu tiên.

1. ### <a name="_toc154182294"></a>***Những thách thức của Continual Learning***
Học tập liên tục đã được áp dụng trong công nghiệp với thành công lớn. Tuy nhiên, nó có ba thách thức lớn mà các công ty cần phải vượt qua:

- Thách thức truy cập dữ liệu mới

  Nếu chúng ta muốn cập nhật mô hình của mình mỗi giờ. Chúng ta cần đào tạo dữ liệu mỗi giờ. Nhịp độ cập nhật càng ngắn thì thách thức này càng trở nên nghiêm trọng.

- Vấn đề về tốc độ lưu trữ vào kho dữ liệu
- Nhiều công ty lấy dữ liệu đào tạo từ kho dữ liệu của họ như Snowflake hoặc BigQuery. Tuy nhiên, dữ liệu đến từ các nguồn khác nhau được gửi vào kho bằng các cơ chế khác nhau và ở tốc độ khác nhau.
- Một cách tiếp cận phổ biến để giải quyết vấn đề này là lấy dữ liệu trực tiếp từ quá trình vận chuyển theo thời gian thực để đào tạo trước khi đưa vào kho. Điều này đặc biệt mạnh mẽ khi việc vận chuyển thời gian thực được nối vào một kho feature. 
- Vấn đề tốc độ ghi nhãn
- Chúng ta có thể thực hiện tính toán nhãn theo đợt. Các công việc hàng loạt này thường chạy định kỳ trên dữ liệu đã được gửi vào kho dữ liệu. Do đó, tốc độ ghi nhãn là một hàm của cả tốc độ lưu trữ dữ liệu và nhịp độ của công việc tính toán nhãn.
- Tương tự như giải pháp ở trên, một cách tiếp cận phổ biến để tăng tốc độ ghi nhãn là tính toán nhãn trực tiếp từ việc vận chuyển (sự kiện) thời gian thực. Tính toán phát trực tuyến này có những thách thức riêng.
- Thách thức đánh giá
- Việc áp dụng continual learning như một phương pháp thực hành có nguy cơ dẫn đến những thất bại thảm khốc của mô hình. Việc cập nhật mô hình càng thường xuyên thì càng có nhiều cơ hội để mô hình thất bại.
- Ngoài ra, việc học hỏi liên tục sẽ mở ra cơ hội cho các cuộc tấn công đối nghịch phối hợp nhằm đầu độc các mô hình.
- Điều này có nghĩa là việc thử nghiệm các mô hình trước khi triển khai chúng cho nhiều đối tượng hơn là rất quan trọng.
- Thách thức mở rộng quy mô dữ liệu
- Tính toán feature thường yêu cầu chia tỷ lệ. Chia tỷ lệ yêu cầu quyền truy cập vào số liệu thống kê dữ liệu toàn cầu như tối thiểu, tối đa, trung bình và phương sai.
- Nếu chúng ta đang sử dụng phương pháp Stateful traning, số liệu thống kê toàn cầu phải xem xét cả dữ liệu trước đó đã được sử dụng để huấn luyện mô hình cộng với dữ liệu mới đang được sử dụng để làm mới mô hình. Việc theo dõi số liệu thống kê toàn cầu trong trường hợp này có thể khó khăn.
- Một kỹ thuật phổ biến để thực hiện việc này là tính toán hoặc ước tính các thống kê này tăng dần khi quan sát dữ liệu mới (ngược lại với việc tải toàn bộ tập dữ liệu vào thời gian đào tạo và tính toán từ đó).
- Thách thức thuật toán
- Thử thách này thể hiện khi sử dụng một số loại thuật toán nhất định và muốn cập nhật chúng thật nhanh (ví dụ: mỗi giờ).
- Các thuật toán được đề cập là những thuật toán mà theo thiết kế, dựa vào việc có quyền truy cập vào tập dữ liệu đầy đủ để được đào tạo. Ví dụ: các mô hình matrix-based, dimensionality reduction-based and tree-based. Những loại mô hình này không thể được huấn luyện tăng dần bằng dữ liệu mới như neural network hoặc các mô hình weight-based khác có thể.
- Thử thách chỉ xảy ra khi bạn cần cập nhật chúng thật nhanh vì bạn không thể chờ thuật toán xem hết toàn bộ tập dữ liệu.

1. ### <a name="_toc154182295"></a>***Bốn giai đoạn của Continual Learning***
- Giai đoạn 1: Các mô hình chỉ được đào tạo lại (retrained) khi đáp ứng hai điều kiện: (1) hiệu suất của mô hình đã suy giảm đến mức hiện tại nó gây hại nhiều hơn là có lợi, (2) Chúng ta có thời gian để cập nhật mô hình.
- Giai đoạn 2:
- Giai đoạn này thường xảy ra khi các mô hình chính của một miền đã được phát triển và do đó ưu tiên không còn là tạo các mô hình mới mà là duy trì và cải thiện các mô hình hiện có. 
- Tần suất đào tạo lại ở giai đoạn này thường dựa trên “gut feeling” (cảm giác ruột thịt).
- Điểm uốn giữa giai đoạn 1 và giai đoạn 2 thường là một tập lệnh được viết để chạy quá trình stateless retraining theo định kỳ. Việc viết tập lệnh này có thể rất dễ hoặc rất khó tùy thuộc vào số lượng phần phụ thuộc cần được phối hợp để đào tạo lại một mô hình.
- Các bước quan trọng của tập lệnh này:
- Kéo dữ liệu (full data).
- Downsample hoặc unsample dữ liệu nếu cần thiết.
- Trích xuất feature.
- Xử lý và/hoặc chú thích nhãn để tạo dữ liệu đào tạo.
- Bắt đầu quá trình đào tạo.
- Đánh giá mô hình mới.
- Triển khai nó (deploy).
- Giai đoạn 3:

  Để đạt được điều này, cần phải cấu hình lại tập lệnh của mình và một cách để theo dõi dữ liệu và dòng dõi mô hình. Một ví dụ về phiên bản dòng mô hình đơn giản.

- V1 và V2 là hai kiến ​​trúc mô hình khác nhau cho cùng một vấn đề.
- V1.2 so với V2.3 có nghĩa là kiến ​​trúc mô hình V1 đang ở lần lặp thứ 2 của quá trình đào tạo lại không trạng thái hoàn toàn và V2 đang ở lần lặp thứ 3.
- V1.2.12 so với V2.3.43 có nghĩa là đã có 12 khóa đào tạo trạng thái được thực hiện trên V1.2 và 43 khóa đào tạo được thực hiện trên V2.3.
- Chúng ta có thể sẽ cần sử dụng điều này cùng với các kỹ thuật lập phiên bản khác như lập phiên bản dữ liệu để theo dõi bức tranh đầy đủ về cách các mô hình đang phát triển.
- Giai đoạn 4:
- Trong giai đoạn này, phần lịch trình cố định của các giai đoạn trước được thay thế bằng một số cơ chế kích hoạt đào tạo lại. Các tác nhân kích hoạt có thể là:
- **Time-based** (Dựa trên thời gian)
- **Dựa trên hiệu suất:** Ví dụ: hiệu suất đã giảm xuống dưới x%
- **Dựa trên khối lượng:** Tổng lượng dữ liệu được dán nhãn tăng 5%
- **Dựa trên sự trôi dạt (Drift-based):** ví dụ: khi phát hiện sự thay đổi phân phối dữ liệu "chính".

1. ### <a name="_toc154182296"></a>***Đo lường giá trị của độ mới dữ liệu***
Một cách để định lượng giá trị của dữ liệu mới hơn là huấn luyện cùng một kiến ​​trúc mô hình với dữ liệu từ 3 khoảng thời gian khác nhau, sau đó kiểm tra từng mô hình dựa trên dữ liệu được gắn nhãn hiện tại.

Ví dụ nếu bạn phát hiện ra rằng việc để mô hình cũ trong 3 tháng sẽ gây ra sự khác biệt 10% về độ chính xác của dữ liệu thử nghiệm hiện tại và 10% là không thể chấp nhận được, thì bạn cần đào tạo lại sau chưa đầy 3 tháng.

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.011.png)

<a name="_toc154182271"></a>Hình 10. Ví dụ về tập dữ liệu nhiều tháng

Hình ảnh này hiển thị các ví dụ về tập dữ liệu nhiều tháng nhưng trường hợp sử dụng của bạn có thể yêu cầu các nhóm thời gian chi tiết hơn như tuần, ngày hoặc thậm chí vài giờ.

1. ### <a name="_toc154182297"></a>***Lời khuyên thiết thực về cách triển khai continual learning***
Việc triển khai học tập liên tục trong học máy bao gồm việc điều chỉnh các mô hình hiện có hoặc thiết kế các thuật toán mới có thể học và thích ứng với các nhiệm vụ hoặc phân phối dữ liệu mới mà không quên kiến ​​thức đã biết trước đó. Dưới đây là các bước và chiến lược để thực hiện việc học tập liên tục:

- Quản lý dữ liệu (Data Management)
- Thiết lập hệ thống quản lý dữ liệu để xử lý các luồng dữ liệu hoặc tác vụ đến.
- Lưu trữ dữ liệu trong quá khứ và làm cho nó có thể truy cập được để cập nhật mô hình.
- Kỹ thuật chính quy hoá (Regularization Techniques)
- Sử dụng các kỹ thuật chính quy hóa để bảo vệ các tham số mô hình quan trọng liên quan đến các tác vụ trước đó.
- Các ví dụ bao gồm Hợp nhất Trọng lượng Đàn hồi (Elastic Weight Consolidation EWC), Thông minh Synaptic (Synaptic Intelligence SI) và chính quy hóa dựa trên đường dẫn (path-based regularization).
- Học trực tuyến (Online Learning)
- Triển khai học tập trực tuyến, trong đó mô hình cập nhật liên tục khi có dữ liệu mới.
- Sử dụng các cập nhật nhỏ hoặc cập nhật gia tăng để thích ứng với thông tin mới.
- Phát lại bộ nhớ (Memory Replay)
- Sử dụng cơ chế phát lại bộ nhớ để lưu trữ và phát lại định kỳ các trải nghiệm trong quá khứ.
- Phát lại dữ liệu cũ giúp mô hình lưu giữ kiến ​​thức về các nhiệm vụ trước đó.
- Chuyển tiếp học tập (Transfer Learning)
- Sử dụng phương pháp học chuyển giao bằng cách khởi tạo các mô hình có trọng số được huấn luyện trước từ các nhiệm vụ hoặc miền liên quan.
- Tinh chỉnh mô hình về các nhiệm vụ mới để điều chỉnh nó một cách hiệu quả.
- Sửa đổi kiến trúc (Architecture Modifications)
- Thử nghiệm các sửa đổi kiến ​​trúc cho phép mô hình thích ứng và mở rộng khi có nhiệm vụ mới.
- Mạng lưới thần kinh tiến bộ, kiến ​​trúc mô-đun và mô hình có thể mở rộng là những ví dụ.
- Đánh giá thường xuyên (Regular Evaluation)
- Liên tục đánh giá hiệu quả hoạt động của mô hình trên cả nhiệm vụ mới và cũ.
- Sử dụng các số liệu đánh giá thích hợp, chẳng hạn như độ chính xác trung bình trên tất cả các nhiệm vụ (MAOT), để theo dõi tiến độ.
- Tỷ lệ quên năng động (Dynamic Forgetting Rate)
- Triển khai các chiến lược hoặc tỷ lệ quên linh hoạt cho mô hình để kiểm soát tốc độ quên thông tin cũ.
- Làm cho quá trình quên thích ứng với tầm quan trọng của các nhiệm vụ trong quá khứ.
- Siêu học tập (Meta-Learning)
- Khám phá các kỹ thuật siêu học tập, trong đó mô hình học cách thích ứng nhanh với các nhiệm vụ mới bằng cách đào tạo các nhiệm vụ khác nhau.
- Khái niệm phát hiện trôi dạt (Concept Drift Detection)
- Phát triển các cơ chế để phát hiện sự trôi dạt khái niệm hoặc những thay đổi trong phân phối dữ liệu.
- Cập nhật mô hình kích hoạt khi phát hiện sai lệch khái niệm quan trọng.
- Nhãn nhiệm vụ (Task Labels)
- Sử dụng nhãn nhiệm vụ hoặc siêu thông tin để hướng dẫn quá trình học tập của mô hình nếu có.
- Thông tin cụ thể về nhiệm vụ có thể giúp mô hình giữ lại hoặc quên thông tin một cách có chọn lọc.
- Bảo trì thường xuyên (Regular Maintenance)
- Các mô hình học tập liên tục cần được bảo trì và giám sát thường xuyên.
- Cập nhật và tinh chỉnh các mô hình khi có dữ liệu mới hoặc khi môi trường thay đổi.
- Cân bằng dữ liệu (Data Balancing)
- Giải quyết các vấn đề mất cân bằng dữ liệu có thể phát sinh khi có nhiệm vụ hoặc luồng dữ liệu mới.
- Đảm bảo rằng mô hình không khớp quá mức với dữ liệu gần đây nhất.
- Bộ dữ liệu và nhiệm vụ điểm chuẩn (Benchmark Datasets and Tasks)
- Đánh giá các thuật toán học tập liên tục của bạn trên các tập dữ liệu và nhiệm vụ tiêu chuẩn để so sánh hiệu suất của chúng với các phương pháp hiện có.

1. ### <a name="_toc154182298"></a>***Khi nào nên lặp lại mô hình***
Hầu phần này cho đến nay em đều đề cập đến việc cập nhật mô hình với dữ liệu mới (tức là lặp lại dữ liệu). Tuy nhiên, trong thực tế, đôi khi chúng ta cũng có thể cần thay đổi kiến ​​trúc mô hình của mình (tức là lặp lại mô hình). Dưới đây là một số gợi ý về thời điểm nên và không nên xem xét việc lặp lại mô hình.

- Nếu chúng ta tiếp tục giảm kích hoạt đào tạo lại việc lặp lại dữ liệu và không thu được nhiều lợi ích, có lẽ bạn nên đầu tư vào việc tìm kiếm một mô hình tốt hơn.
- Nếu việc thay đổi sang kiến ​​trúc mô hình lớn hơn yêu cầu sức mạnh tính toán 100X sẽ cải thiện hiệu suất 1%, nhưng việc giảm thời gian kích hoạt đào tạo lại xuống còn 3 giờ cũng giúp bạn tăng hiệu suất 1% với sức mạnh tính toán 1X, hãy ưu tiên việc lặp lại dữ liệu hơn việc lặp lại mô hình.
- Câu hỏi "khi nào thực hiện lặp lại mô hình và lặp lại dữ liệu" vẫn chưa được nghiên cứu trả lời rõ ràng cho tất cả các nhiệm vụ. Chúng ta sẽ cần phải chạy thử nghiệm nhiệm vụ cụ thể của mình để tìm ra thời điểm thực hiện việc đó.
  1. ## <a name="_toc154182299"></a>**Test Production**
Sau khi mô hình được cập nhật, nó không thể được phát hành một cách mù quáng vào sản xuất. Nó cần phải được thử nghiệm để đảm bảo rằng nó an toàn và tốt hơn so với mô hình hiện tại đang được sản xuất.

Để kiểm tra đầy đủ các mô hình của chúng ta trước khi phổ biến rộng rãi, chúng ta cần cả đánh giá ngoại tuyến trước khi triển khai và thử nghiệm trong sản xuất. Chỉ đánh giá ngoại tuyến là không đủ.

Lý tưởng nhất là chúng ta đưa ra nhiều quy trình rõ ràng về cách đánh giá các mô hình: thử nghiệm nào sẽ chạy, ai thực hiện chúng và các ngưỡng áp dụng để thúc đẩy mô hình lên giai đoạn tiếp theo. Tốt nhất là các quy trình đánh giá này được tự động hóa và khởi động khi có bản cập nhật mô hình mới. Việc thăng cấp giai đoạn cần được xem xét tương tự như cách đánh giá CI/CD trong công nghệ phần mềm.
1. ### <a name="_toc154182300"></a>***Đánh giá ngoại tuyến trước khi triển khai.***
Hai cách phổ biến nhất là (1) Sử dụng **phần tách thử nghiệm** (**test-split**) để so sánh với đường cơ sở và (2) **chạy thử nghiệm ngược** (**backtests**)

- Test-split thường ở dạng tĩnh để bạn có điểm chuẩn đáng tin cậy để so sánh nhiều mô hình. Điều này cũng có nghĩa là hiệu suất tốt trên phần test-split tĩnh cũ không đảm bảo hiệu suất tốt trong điều kiện phân phối dữ liệu hiện tại trong sản xuất.
- Backtest là ý tưởng sử dụng dữ liệu được gắn nhãn mới nhất mà mô hình chưa thấy trong quá trình đào tạo để kiểm tra hiệu suất (ví dụ: nếu đã sử dụng dữ liệu của ngày cuối cùng, hãy sử dụng dữ liệu của giờ cuối cùng để kiểm tra lại).
  1. ### <a name="_toc154182301"></a>***Thử nghiệm trong chiến lược sản xuất***
     1. #### Triển khai bóng (Shadow Deployment)
- **Trực giác (Intuition)**: Triển khai mô hình người thách đấu song song với mô hình nhà vô địch hiện có. Gửi mọi yêu cầu đến cả hai mô hình nhưng chỉ phục vụ suy luận của mô hình vô địch. Ghi lại các dự đoán cho cả hai mô hình để so sánh chúng.
- **Ưu điểm**:
- Đây là cách an toàn nhất để triển khai mô hình. Ngay cả khi mô hình mới có lỗi, dự đoán sẽ không được cung cấp.
- Rất dễ hiểu
- Thử nghiệm sẽ thu thập đủ dữ liệu để đạt được ý nghĩa thống kê nhanh hơn tất cả các chiến lược khác vì tất cả các mô hình đều nhận được lưu lượng truy cập đầy đủ.
- **Nhược điểm**:
- Không thể sử dụng kỹ thuật này khi đo lường hiệu suất của mô hình phụ thuộc vào việc quan sát cách người dùng tương tác với các dự đoán. Ví dụ: các dự đoán từ mô hình đề xuất bóng tối (shadow) sẽ không được cung cấp nên chúng ta sẽ không thể biết liệu người dùng có nhấp vào chúng hay không.
- Kỹ thuật này tốn kém khi chạy vì nó tăng gấp đôi số lượng dự đoán và do đó số lượng tính toán cần thiết.
  1. #### A/B Testing
- **Intuition:** Triển khai mô hình người thách thức cùng với mô hình quán quân (mô hình A) và định tuyến phần trăm lưu lượng truy cập (*percentage of traffic)* đến người thách thức (mô hình B). Dự đoán từ người thách đấu được hiển thị cho người dùng. Sử dụng tính năng theo dõi và phân tích dự đoán trên cả hai mô hình để xác định xem thành tích của người thách đấu có tốt hơn về mặt thống kê so với nhà vô địch hay không
- Một số trường hợp sử dụng không phù hợp với ý tưởng phân chia lưu lượng truy cập và có nhiều mô hình cùng một lúc. Trong những trường hợp này, thử nghiệm A/B có thể được thực hiện bằng cách thực hiện phân chia theo thời gian: mô hình A ngày hôm nay, mô hình B ngày hôm sau.
- Việc phân chia lưu lượng truy cập phải là một thử nghiệm thực sự ngẫu nhiên. Nếu có bất kỳ sự thiên vị lựa chọn nào về việc ai sẽ nhận được mô hình A và mô hình B (như người dùng máy tính để bàn nhận được A và thiết bị di động nhận được B), kết luận của bạn sẽ không chính xác.
- Thí nghiệm phải chạy đủ lâu để thu thập đủ mẫu nhằm đạt được độ tin cậy thống kê đủ lớn về sự khác biệt.
- Ý nghĩa thống kê không phải là bằng chứng sai lệch (đó là lý do tại sao nó có độ tin cậy). Nếu không có sự khác biệt thống kê giữa A và B, chúng ta có thể sử dụng cả hai.

- **Ưu điểm:**
- Vì dự đoán được cung cấp cho người dùng nên kỹ thuật này cho phép bạn nắm bắt đầy đủ cách người dùng phản ứng với các mô hình khác nhau.
- Thử nghiệm A/B rất dễ hiểu và có rất nhiều thư viện cũng như tài liệu xung quanh nó.
- Việc chạy này rẻ vì chỉ có một dự đoán cho mỗi yêu cầu.

- **Nhược điểm:**
- Nó kém an toàn hơn so với triển khai bóng. Bạn muốn có một số đánh giá ngoại tuyến mạnh mẽ hơn đảm bảo rằng mô hình của bạn sẽ không thất bại thảm hại vì bạn sẽ đưa lưu lượng truy cập thực sự qua mô hình đó.
- Chúng ta sẽ không cần phải xem xét các trường hợp đặc biệt phát sinh từ các yêu cầu suy luận song song cho các chế độ dự đoán trực tuyến (xem nhược điểm của việc triển khai bóng).
  1. #### Canary Release (Phát hành Canary)
- **Intuition:** Triển khai người thách đấu và nhà vô địch cạnh nhau nhưng bắt đầu với người thách đấu không tham gia giao thông. Từ từ di chuyển lưu lượng truy cập từ nhà vô địch đến kẻ thách thức. Theo dõi số liệu hiệu suất của người thách thức, nếu chúng trông ổn, hãy tiếp tục cho đến khi tất cả lưu lượng truy cập đều thuộc về người thách thức.
- Các bản phát hành Canary có thể được kết hợp với thử nghiệm A/B để đo lường nghiêm ngặt sự khác biệt về hiệu suất.
- Các bản phát hành Canary cũng có thể được chạy ở "chế độ YOLO", trong đó chúng ta quan sát sự khác biệt về hiệu suất.

- **Ưu điểm:**
- Dễ hiểu
- Đơn giản.
- Vì dự đoán của người thách thức sẽ được đưa ra nên bạn có thể sử dụng điều này với các mô hình yêu cầu tương tác của người dùng để nắm bắt hiệu suất.
- So với việc triển khai bóng thì việc chạy sẽ rẻ hơn.
- Nếu kết hợp với thử nghiệm A/B, nó cho phép thay đổi linh hoạt lượng lưu lượng truy cập mà mỗi mô hình đang sử dụng.

- **Nhược điểm:**
- Nó mở ra khả năng không nghiêm ngặt trong việc xác định sự khác biệt về hiệu suất.
- Nếu việc xả thải không được giám sát cẩn thận, tai nạn có thể xảy ra. Đây được cho là lựa chọn kém an toàn nhất nhưng lại rất dễ bị rollback.

1. #### Interleaving Experiments (Thí nghiệm xen kẽ)
- **Intuition:** Trong thử nghiệm A/B, một người dùng sẽ nhận được dự đoán từ mô hình A hoặc mô hình B. **Interleaving**, một người dùng sẽ nhận được dự đoán xen kẽ từ cả mô hình A và mô hình B. Sau đó, chúng tôi theo dõi hiệu quả hoạt động của từng mô hình bằng cách đo lường mức độ ưu tiên của người dùng với từng mô hình dự đoán của mô hình (ví dụ: người dùng nhấp nhiều hơn vào đề xuất từ ​​mô hình B).
- Nhiệm vụ đề xuất là trường hợp sử dụng điển hình của việc xen kẽ. Không phải tất cả các nhiệm vụ đều phù hợp với chiến lược này.

![](Aspose.Words.382626bd-ff96-4a86-8a33-694b856e3d57.012.png)

<a name="_toc154182272"></a>Hình 11. Hình minh hoạ về thử nghiệm **Interleaving** so với **A/B Testing**

- **Ưu điểm**: 
- Sử dụng việc xen kẽ xác định mô hình tốt nhất một cách đáng tin cậy với kích thước mẫu nhỏ hơn đáng kể so với thử nghiệm A/B truyền thống.
- Ngược lại với việc triển khai theo dõi, chiến lược này cho phép bạn nắm bắt cách người dùng hành xử trái với dự đoán của bạn (vì dự đoán được đưa ra).
- **Nhược điểm**:
- Việc triển khai phức tạp hơn thử nghiệm A/B.
- Nó tăng gấp đôi sức mạnh tính toán cần thiết vì mọi yêu cầu đều nhận được dự đoán từ nhiều mô hình.
- Nó không thể được sử dụng cho tất cả các loại nhiệm vụ. Ví dụ: nó hoạt động cho các nhiệm vụ xếp hạng/đề xuất nhưng nó không có ý nghĩa đối với các nhiệm vụ hồi quy.
- Nó không dễ dàng mở rộng quy mô thành một số lượng lớn các mô hình thách thức.
  1. #### Bandits (kẻ cướp)
- **Intuition**: Là một thuật toán theo dõi hiệu suất hiện tại của từng biến thể mô hình và đưa ra quyết định linh hoạt đối với mọi yêu cầu về việc nên sử dụng mô hình có hiệu suất cao nhất cho đến nay (tức là khai thác kiến ​​thức hiện tại).
- **Bandits** thêm một khái niệm khác vào quyết định sử dụng mô hình nào: chi phí cơ hội (**Opportunity cost**).
- Có rất nhiều thuật toán Bandits. Đơn giản nhất được gọi là epsilon-greedy. Hai công cụ mạnh mẽ và phổ biến nhất là Thompson Sampling và Upper Confidence Bound (UCB).

- **Ưu điểm:**
- **Bandits** cần ít mất dữ liệu hơn thử nghiệm A/B để xác định mô hình nào tốt hơn. Một ví dụ được đưa ra là đến 630K mẫu để đạt độ tin cậy 95% khi thử nghiệm A/B và chỉ mất 12K với bandits.
- Bandits sử dụng dữ liệu hiệu quả hơn đồng thời giảm thiểu chi phí cơ hội. Trong nhiều trường hợp bandits được coi là tối ưu.
- So với thử nghiệm A/B, bandits an toàn hơn vì nếu một mô hình thực sự tệ, thuật toán sẽ chọn nó ít thường xuyên hơn. Ngoài ra, tốc độ hội tụ sẽ nhanh hơn nên chúng ta có thể loại bỏ kẻ thách thức xấu một cách nhanh chóng.

- **Nhược điểm:**
- So với tất cả các chiến lược khác, **bandits** khó thực hiện hơn nhiều do cần phải truyền phản hồi vào thuật toán một cách liên tục.

# **<a name="_toc154182302"></a>TÀI LIỆU THAM KHẢO**
Tiếng Việt

VIBLO, *Optimizer- Hiểu sâu về các thuật toán tối ưu,* <https://s.net.vn/vicv>

Tiếng Anh

Analytics Vidhya, *A Comprehensive Guide on Optimizers in Deep Learning, <https://s.net.vn/Cl1o>*

Github, Serodriguez68, *Continual learning and test in production*, <https://s.net.vn/CQJU>

Spot Intelligence, *Continual Learning Made Simple, How To Get Started & Top 4 Models,* <https://s.net.vn/36QX>

Linkedin, *Machine Learning Optimization Techniques*, <https://s.net.vn/RrEl>


