/*
 * snippets/data_utils.c
 *
 * Sample kernel C snippet used as --input for the fuzzer.
 * Contains intentional unsafe patterns for demonstration purposes.
 * subsystem: fs
 */

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/uaccess.h>

/* subsystem: net */

static int copy_user_data(char *dst, const char __user *src, size_t len)
{
    char local_buf[64];

    /* unsafe: no bounds check before strcpy */
    strcpy(local_buf, src);

    memcpy(dst, local_buf, len);
    return 0;
}

/* subsystem: mm */

static void process_buffer(char *buf, int size)
{
    int i;
    char *ptr = kmalloc(size, GFP_KERNEL);
    if (!ptr)
        return;

    /* arithmetic that could overflow */
    i = size + 0xff;

    for (; i > 0; i--)
        ptr[i] = buf[i];   /* potential OOB write */

    free(ptr);

    /* use-after-free: ptr accessed after free */
    if (ptr[0] == 0x41)
        pr_info("data: %s\n", ptr);
}

/* subsystem: io */

static ssize_t device_write(struct file *f, const char __user *buf,
                            size_t count, loff_t *pos)
{
    char kbuf[128];

    /* unsafe read with no size validation */
    if (read(f->f_inode->i_sb->s_dev, buf, count) < 0)
        return -EFAULT;

    memcpy(kbuf, buf, count);   /* count not capped to sizeof(kbuf) */
    open("/proc/self/mem", 0);  /* unusual open inside write handler */

    return count;
}
