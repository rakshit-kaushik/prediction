# Dashboard Input Methods - Quick Guide

The enhanced dashboard now supports **3 ways** to specify which market to analyze:

## Input Method 1: Polymarket URL (Recommended)

**Best for**: Easiest method - just copy-paste from your browser

**How to use:**
1. Go to Polymarket.com and find your market
2. Copy the full URL from your browser address bar
3. Select "Polymarket URL" in the dashboard
4. Paste the URL
5. Dashboard automatically extracts the market slug

**Example URLs:**
```
https://polymarket.com/event/fed-decision/fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting

https://polymarket.com/event/super-bowl-champion-2026-731

https://polymarket.com/market/trump-wins-2024
```

**Advantages:**
- No manual slug extraction needed
- Works with any Polymarket URL format
- Instant validation (green checkmark if slug extracted)

---

## Input Method 2: Market Slug

**Best for**: When you already know the market slug

**How to use:**
1. Select "Market Slug" in the dashboard
2. Type or paste the market slug
3. Click "Verify Market" to check if it exists (optional)

**Example slugs:**
```
fed-decreases-interest-rates-by-25-bps-after-december-2025-meeting

trump-wins-2024-presidential-election

super-bowl-champion-2026
```

**Finding the slug:**
- Look at Polymarket URL: `polymarket.com/event/[event]/[MARKET-SLUG]`
- Or use the URL input method instead

**Advantages:**
- Direct input if you have the slug
- Can verify via Gamma API
- Shows current Yes/No prices if available

---

## Input Method 3: Token ID (Advanced)

**Best for**: When slug lookup fails or you know the token ID

**How to use:**
1. Select "Token ID" in the dashboard
2. Enter the Polymarket token ID
3. Click "Verify Token ID" to check if data exists (optional)

**Example token IDs:**
```
21742633143463906290569050155826241533067272736897614950488156847949938836455

69236923620077691027083946871148646972011131466059644796654161903044970987404
```

**Finding the token ID:**
- Inspect Polymarket page network requests
- Look for `clobTokenIds` in API responses
- Check market metadata JSON files

**Advantages:**
- Bypasses slug lookup (fastest method)
- Works when Gamma API doesn't have the market
- Direct access to Dome API data

---

## Comparison Table

| Feature | URL | Slug | Token ID |
|---------|-----|------|----------|
| **Ease of use** | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐⭐ Easy | ⭐⭐ Advanced |
| **Works for new markets** | ✅ Yes | ⚠️ Maybe | ✅ Yes |
| **Auto-extracts slug** | ✅ Yes | N/A | N/A |
| **Shows prices** | ✅ Yes | ✅ Yes | ❌ No |
| **Verification** | ✅ Instant | ✅ Via Gamma API | ✅ Via Dome API |
| **Fastest** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Troubleshooting by Input Method

### URL Method Issues

**❌ "Could not extract slug from URL"**
- Check URL format - should be from polymarket.com
- Try copying URL again from browser
- Make sure URL contains `/event/` or `/market/`

**Solution:** Try copying the URL when you're on a specific market outcome page

---

### Slug Method Issues

**⚠️ "Market not found in Gamma API"**
- This is NORMAL for newer markets (2024+)
- Gamma API only has historical markets
- You can still proceed - Dome API will be used

**❌ "Could not find token ID for market slug"**
- Slug might be misspelled
- Market might not exist
- Try using the URL method instead

**Solutions:**
1. Click on the specific outcome you want on Polymarket
2. Copy that URL and use URL input method
3. Or find the token ID and use Token ID method

---

### Token ID Method Issues

**❌ "Could not verify token ID"**
- Token ID might be incorrect
- No data available for this token
- Check DOME_API_KEY is set in .env file

**Solutions:**
1. Double-check the token ID
2. Verify .env file has DOME_API_KEY
3. Try using URL or Slug method

---

## Recommended Workflow

### For Quick Analysis:
1. **Use URL method** (copy from browser)
2. Select dates
3. Click "Run Analysis Pipeline"

### For Repeated Analysis:
1. **Use Token ID method** (faster, no lookup)
2. Save token IDs for markets you analyze frequently
3. Skip verification step

### When Slug Lookup Fails:
1. Start with URL method
2. If that extracts a slug, click "Verify"
3. If verification shows token ID, save it for future use
4. Next time, use Token ID method directly

---

## Examples by Use Case

### Use Case: New Market (2024+)

**Problem:** Newer markets not in Gamma API

**Solution:**
```
Method 1 (Recommended):
- Input Method: "Polymarket URL"
- Paste: https://polymarket.com/event/something/market-slug-here
- Auto-extracts slug ✅
- Run pipeline (will find token via Dome API)

Method 2 (If you know token):
- Input Method: "Token ID"
- Enter: 21742...
- Skip verification
- Run pipeline immediately
```

---

### Use Case: Analysis of Multiple Markets

**Problem:** Need to analyze 5+ markets quickly

**Solution:**
```
First time:
1. Use URL method for each market
2. Click "Verify" to get token ID
3. Save token IDs to a text file

Future analyses:
1. Use Token ID method
2. Paste saved token ID
3. Run pipeline (no lookup delay)
```

---

### Use Case: Slug Not Working

**Problem:** Gamma API can't find the slug

**Solution:**
```
Step 1: Go to Polymarket page
Step 2: Click on the specific "Yes" or "No" outcome you want
Step 3: URL changes to include market slug
Step 4: Copy that new URL
Step 5: Use URL input method in dashboard
Step 6: Dashboard extracts the correct slug
```

---

## Quick Reference: When to Use Which Method

| Situation | Best Input Method |
|-----------|------------------|
| First time analyzing a market | **URL** |
| Repeated analysis of same market | **Token ID** |
| Sharing analysis steps with others | **URL** or **Slug** |
| Slug lookup failing | **Token ID** |
| Want to see current prices | **URL** or **Slug** |
| Fastest possible run | **Token ID** |
| Market from 2020-2021 | **URL** or **Slug** |
| Market from 2024+ | **URL** or **Token ID** |

---

## Pro Tips

1. **Save Token IDs**: After first successful run, save the token ID for future analyses

2. **Bookmark Market URLs**: Keep URLs of markets you analyze frequently

3. **Skip Verification**: If you know the market exists, skip the verify button and go straight to "Run Pipeline"

4. **Use Token ID for Batch Processing**: If analyzing multiple markets, collect token IDs first, then batch process

5. **Check .env**: Make sure `DOME_API_KEY` is set before using Token ID verification

---

## Error Messages Explained

| Error | Meaning | Solution |
|-------|---------|----------|
| "Could not extract slug" | URL format not recognized | Copy URL from specific market outcome page |
| "Market not found in Gamma API" | Normal for new markets | Proceed anyway - uses Dome API |
| "Could not find token ID" | Slug lookup failed | Use URL method or get token ID manually |
| "Could not verify token ID" | Token invalid or no data | Check token ID and DOME_API_KEY |
| "Please enter a token ID" | Token ID field empty | Enter a token ID or switch methods |

---

**Last Updated:** November 2025
**Dashboard Version:** 2.0 (Enhanced Input Methods)
