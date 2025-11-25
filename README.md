# master-automating-pose-estimation

## Short guide: SSH access to GitHub

Generate a key, add it to GitHub, test, then use it to clone/pull/push.

```bash
# 1) Generate key
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2) Start agent and add key
eval "$(ssh-agent -s)"

# 3) Show public key (copy this)
cat ~/.ssh/id_ed25519.pub
```

On GitHub: **Settings → SSH and GPG keys → New SSH key**, paste the copied key, save.

```bash
# 4) Test
ssh -T git@github.com

# 5) Clone private repo via SSH
git clone git@github.com:owner/repo.git

# 6) Normal workflow
git pull
git push
```

**References**
- GitHub Docs: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
- GitHub Docs: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account