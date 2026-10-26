# 部署: 让其他电脑访问 HOI → G1 网页

计算都在这台 4070 上跑; 其他电脑只用浏览器打开网页, 不需要装任何东西。
`hoi_app.py` 的启动参数:

| 参数 | 说明 |
|---|---|
| `--host 0.0.0.0` | 允许局域网访问 (默认 `127.0.0.1` 只有本机能开) |
| `--port 7860` | 端口 |
| `--auth user:pass` | 登录账号密码, 也可用环境变量 `HOI_APP_AUTH`; 对外开放 (`--host` 非本机或 `--share`) 时必须设置, 否则拒绝启动 |
| `--share` | Gradio 临时公网链接 (72 小时) |

任务排队执行 (一次一个, 只有一张显卡), 上传文件上限 500 MB。

## 1. 局域网访问
```bash
conda activate g1demo
export HOI_APP_AUTH='g1:<密码>'
python hoi_app.py --host 0.0.0.0
hostname -I                    # 本机局域网 IP, 例如 192.168.1.114
sudo ufw status                # 如果是 active:
sudo ufw allow 7860/tcp
```
其他电脑 (同一 Wi-Fi) 打开 `http://<IP>:7860`, 输入账号密码。

## 2. 不在同一网络
**A. Tailscale (推荐: 私有, 不过期)**
```bash
curl -fsSL https://tailscale.com/install.sh | sh   # 已安装可跳过
sudo tailscale up              # 按提示用浏览器登录
tailscale ip -4                # 例如 100.x.y.z
```
其他电脑也装 Tailscale 并登录同一个账号, 打开 `http://100.x.y.z:7860`。服务端仍用 `--host 0.0.0.0`。

**B. Gradio 临时公网链接 (只用于临时演示, 用完就关)**
```bash
python hoi_app.py --share --auth 'g1:<密码>'
```
终端打印 `https://xxxx.gradio.live`, 拿到链接的人都能访问, 所以必须设密码。

## 3. 开机自动启动 (systemd 用户服务)
`~/.config/systemd/user/hoi-app.service` (密码只写在这里, 不要提交到 git; 建议 `chmod 600`):
```ini
[Unit]
Description=HOI to G1 web app
After=network-online.target

[Service]
WorkingDirectory=%h/code/intern/newrobot/robot
Environment=HOI_APP_AUTH=g1:<密码>
Environment=MUJOCO_GL=egl
ExecStart=%h/miniconda3/envs/g1demo/bin/python hoi_app.py --host 0.0.0.0
Restart=on-failure

[Install]
WantedBy=default.target
```
```bash
systemctl --user daemon-reload
systemctl --user enable --now hoi-app
loginctl enable-linger $USER   # 没登录桌面也保持运行
journalctl --user -u hoi-app -f
```

## 4. 清理临时文件
每次运行的临时文件在 `/tmp/hoi_app_*`。每天清理一次 (`crontab -e` 加一行):
```
0 4 * * * find /tmp -maxdepth 1 -name 'hoi_app_*' -mtime +1 -exec rm -rf {} +
```
