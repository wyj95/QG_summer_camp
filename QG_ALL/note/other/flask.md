# flask

[toc]

### 修饰器

[python 装饰器详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/87353829)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220722204954.png)

****

### 变量规则

~~~python
@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id
~~~

****

### 转换器

如上所示。int即为转换器，另外还有string，float，path或自己定义的转换器

****

### 构造URL

~~~python
with app.test_request_context():
	print url_for('index')
	print url_for('login')
	print url_for('login', next='/')
	print url_for('profile', username='John Doe')
# 后面两个不是特别了解

Put：
/login
/login?next=/
/user/John%20Doe
~~~

****

### HTPP 方法

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220725185030.png)

****

### 重定向和错误

~~~python
from flask import abort, redirect, url_for

@app.route('/')
def index():
    return redirect(url_for('login'))
# 把用户重定向到其他地方

@app.route('/login')
def login():
    abort(401) # 放弃请求并返回错误代码
    this_is_never_executed()
    
from flask import render_template
# 错误捕获
@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404
~~~

****

## 响应

不懂

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220725195952.png)

****

### 日志记录

~~~python
app.logger.debug('A value for debugging')
app.logger.warning('A warning occurred (%d apples)', 42)
app.logger.error('An error occurred')
~~~

****

### 模板渲染

~~~python
from flask import render_template

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)
# hello.html should be set in the 'templates' having been set in app being build
~~~

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220727004835.png) 

### request

get 在表单中应该有name=  "xx"

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220726195942.png)

可以通过

~~~python
request.path
request.full_path 
request.args # 访问
request.arg.get("username")
# 上只能取到get请求的

# post的内容在请求体内，不在url上
request.form
request.form.get("username")
# post只能用form
~~~

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220726195604.png)

### url_for

~~~python
@app.route('\iuwgbsduiguisdgbuivgb', endpoit="index")
def f():
    pass


url_for("index") == '\...'
~~~

### 重定向

~~~python
redirect("\xxx")
# 可以跑到新的网页去，状态码3xx
~~~



