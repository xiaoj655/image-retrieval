# 微调Resnet101完成图像检索任务
通过在线网站访问试用 https://movie-image-retrieval.vercel.app

## 数据集
从223部经典电影计算平均哈希的方式, 筛选出30万张图片作为数据集

223部电影: https://www.alipan.com/s/HAxoueCChn2

30万张原始图片: https://www.alipan.com/t/u3ovD0eXQKxqD8zkGyHD

## 方法
孪生网络, 对比学习的思路, 把经过图像增强后的图片和原图作为一组图像, 让相似的图像的空间嵌入接近, 不相似的远离

## 效果
![query](query.png)
