# A-Insight Pro 每日定时推送配置

## 方案A：GitHub Actions 云端执行（推荐，关机也能推送）

### 1. 推送代码到 GitHub
```bash
git init
git add -A
git commit -m "A-Insight Pro V4.0"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

### 2. 配置 SMTP 密钥
在 GitHub 仓库 → Settings → Secrets and variables → Actions → New repository secret:

| Secret 名称 | 值 |
|-------------|-----|
| `SMTP_HOST` | smtp.qq.com |
| `SMTP_PORT` | 465 |
| `SMTP_USER` | your_email@qq.com |
| `SMTP_PASS` | QQ邮箱授权码 |
| `TO_EMAIL` | 接收报告的邮箱 |

### 3. 启用工作流
推送后自动生效。每天 8:30 AM (北京时间) 自动运行。

手动测试: GitHub Actions → A-Insight Pro Daily Pipeline → Run workflow

---

## 方案B：Windows 本地定时任务

### 1. 创建计划任务（管理员 PowerShell）
```powershell
$action = New-ScheduledTaskAction -Execute "D:\DProjectsA-Insight-Pro\run_daily.bat" -Argument "-q"
$trigger = New-ScheduledTaskTrigger -Daily -At 8:30AM
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -WakeToRun
Register-ScheduledTask -TaskName "A-Insight-Pro-Daily" -Action $action -Trigger $trigger -Principal $principal -Settings $settings
```

`-WakeToRun` 会让电脑从睡眠中唤醒执行。**但不能从关机状态唤醒**。

### 2. 配置邮件推送
编辑 `config/notify_config.json`：
```json
{
  "smtp_host": "smtp.qq.com",
  "smtp_port": 465,
  "smtp_user": "你的QQ邮箱@qq.com",
  "smtp_pass": "QQ邮箱SMTP授权码(不是密码)",
  "to_email": "接收推送的邮箱",
  "enabled": true
}
```

QQ邮箱授权码获取：设置 → 账户 → POP3/SMTP服务 → 开启 → 生成授权码

### 3. BIOS 设置自动开机（解决关机问题）
- 进入 BIOS → Power Management → Auto Power On
- 设置为每天 8:25 AM 自动开机
- Windows 计划任务 8:30 AM 执行
- 任务完成后自动休眠/关机

配合 `WakeToRun` + BIOS 自动开机 = 关机也能推送

---

## 收到的每日推送内容

```
标题: A-Insight Daily 2026-07-29

内容:
- 市场情绪: 29 (偏谨慎) | 主力: 主力休息
- 信号: 3 Buy | 60 Watch
- 模型偏差: -0.02% | 准确率: 54.4%
- Top 10 股票排名表 (代码/名称/预测/评分/信号)
```

## 测试
```bash
# 本地测试完整流程
run_daily.bat

# 单独测试邮件推送
venv\Scripts\python -m ai.notify
```
