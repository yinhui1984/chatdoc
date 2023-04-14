# chatdoc

针对各类文档使用chatgpt

在使用chatgpt的时候,经常会遇到内容大小超过4097限制 (比如分析一本书), 而无法愉快地玩耍了. 这里的小工具先将文件文件提取再转换成向量, 再使用 `RetrievalQAWithSourcesChain`进行问答

## 使用

先复制`env.config.example` 为`env.conf` 并修改其中的`OPENAI_API_KEY`为你自己的key

### 方式1, 以文件作为内容输入

`python3 chatdoc.py file_url question`

其中file_url可以是pdf和任意文本文件

以一本关于solidity的书籍为例

```shell
python3  ./chatdoc.py /Users/zhouyinhui/blockchain/solidity/1.pdf  explain what is modifier and give me an example
```

![](https://github.com/yinhui1984/imagehosting/blob/main/images/1681448484817104000.png?raw=true)

### 方式2, 以stdin作为内容输入

`somecomment xxx | python3 ./chatdoc.py question  `

```sh
cat chatdoc.py | python3 ./chatdoc.py sumarize the main feature of these code
```

![image](https://github.com/yinhui1984/imagehosting/blob/main/images/1681448840693616000.png?raw=true)



```shell
curl https://raw.githubusercontent.com/yinhui1984/yinhui1984.github.io/main/content/posts/Euler-Finance-Attack.md | python3 ./chatdoc.py list the attack steps
```

![image](https://github.com/yinhui1984/imagehosting/blob/main/images/1681449923750591000.png?raw=true)

